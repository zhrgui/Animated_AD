import argparse
import os
import shutil
import json
import numpy as np
import subprocess
import random
import torchaudio
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier


def crop_audio(input_video, output_video, start_time, end_time):
    """
    Use ffmpeg to trim an audio/video file between start_time and end_time,
    saving the result to output_video without re-encoding.
    """
    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", input_video,
        "-c", "copy",
        output_video
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cosine_similarity(arr1, arr2):
    """
    Compute the cosine similarity between two arrays.
    Returns a value in [-1, 1], or 0.0 if either array is all zeros.
    """
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")
    dot_product = np.dot(arr1, arr2)
    magnitude_arr1 = np.linalg.norm(arr1)
    magnitude_arr2 = np.linalg.norm(arr2)

    if magnitude_arr1 == 0 or magnitude_arr2 == 0:
        return 0.0
    cosine_sim = dot_product / (magnitude_arr1 * magnitude_arr2)
    return cosine_sim


def normalize_vector(vec):
    """
    Safely normalize a single vector to unit length.
    If vec has zero norm, returns vec unchanged.
    """
    norm_val = np.linalg.norm(vec)
    if norm_val > 1e-12:
        return vec / norm_val
    return vec


def verify_features(feature_a, feature_b):
    """
    Compute the cosine similarity between two feature vectors.
    Returns the similarity score and a boolean indicating whether the
    features are considered a match (score >= 0.25).
    """
    score = cosine_similarity(feature_a, feature_b)
    prediction = score >= 0.25
    return score, prediction


def load_audio_feature_bank(example_audio_file):
    """
    Load a hierarchical audio feature bank from a JSON specification.
    """
    audio_feature_bank = {}
    with open(example_audio_file) as infile:
        audio_feature_files = json.load(infile)

    for movie_title, per_movie_feature_files in audio_feature_files.items():
        per_movie_feature_bank = {}
        for character, feature_files in per_movie_feature_files.items():
            feature_list = [np.load(feature_file)['feature'] for feature_file in feature_files]
            per_movie_feature_bank[character] = feature_list
        audio_feature_bank[movie_title] = per_movie_feature_bank
    return audio_feature_bank


def group_speakers(features_list, threshold=0.25):
    """
    Group audio features into speaker clusters based on cosine similarity.
    """
    speakers = []  # list of tuples: (indices in features_list, stacked feature vectors)

    # Iterate through each feature vector and assign it to an existing cluster or create a new one
    for feat_idx, feat_vec in enumerate(features_list):
        feat_vec_2d = np.array(feat_vec).reshape(1, -1)  # ensure 2D shape for similarity computation
        placed = False

        # Compare with each existing cluster
        for spk_i, (indices, spk_vectors) in enumerate(speakers):
            sim_values = cosine_similarity(feat_vec_2d, spk_vectors)
            if np.any(sim_values > threshold):  # check if it matches the cluster
                indices.append(feat_idx)
                speakers[spk_i] = (indices, np.vstack([spk_vectors, feat_vec_2d]))
                placed = True
                break

        # If not similar to any existing cluster, create a new one
        if not placed:
            speakers.append(([feat_idx], feat_vec_2d))

    speaker_dict = {}  # cluster index -> list of feature indices
    centroids = {}     # cluster index -> centroid vector

    # Build final dictionaries for cluster membership and centroids
    for spk_idx, (indices, spk_vectors) in enumerate(speakers):
        speaker_dict[spk_idx] = indices

        # Normalize all vectors in the cluster
        normed_vectors = [normalize_vector(row) for row in spk_vectors]
        normed_vectors = np.array(normed_vectors)

        # Compute centroid as the normalized sum of vectors
        sum_vec = np.sum(normed_vectors, axis=0)
        centroid = normalize_vector(sum_vec).reshape(1, -1)
        centroids[spk_idx] = centroid

    return speaker_dict, centroids


def identify_active_speaker(query_feature, audio_example_bank):
    """
    Identify the most likely active speaker by comparing a query audio feature 
    against a bank of stored character audio features.
    """
    char_to_score = {}

    for char_name, feature_list in audio_example_bank.items():
        cosine_sim = []
        for feature in feature_list:
            score, _ = verify_features(query_feature, feature)
            cosine_sim.append(score)

        cosine_sim.sort(reverse=True)

        if len(cosine_sim) <= 3:
            classification_score = sum(cosine_sim) / len(cosine_sim)
        else:
            classification_score = sum(cosine_sim[:3]) / 3

        char_to_score[char_name] = classification_score

    max_score = 0.
    max_char_name = None
    for char_name, score in char_to_score.items():
        if score >= max_score:
            max_score = score
            max_char_name = char_name

    return {
        'identified_speaker': max_char_name,
        'identified_score': max_score,
    }


def main():
    os.makedirs(args.temp_audio_dir, exist_ok=True)

    # Load the pre-annotated unclassified speech segments
    with open(args.audio_annotation_file, 'r') as infile:
        annotations = json.load(infile)

    # Load the mapping from movie titles to video indices
    with open(args.movie_to_video_file, 'r') as infile:
        movie_video_list = json.load(infile)

    # Load the in-movie audio feature bank from a JSON file
    audio_example_bank = load_audio_feature_bank(args.example_audio_file)

    # Load the crawled actor voice feature bank from a JSON file
    actor_audio_bank = load_audio_feature_bank(args.actor_audio_bank_file)

    def augment_audio_bank(audio_example_bank, actor_audio_bank, k = 30):
        """
        Augment the per-character audio feature bank with additional samples 
        from the corresponding actor's audio bank to ensure a minimum number of features.
        """
        for movie_title, per_movie_audio_bank in audio_example_bank.items():
            for character_name, feature_list in per_movie_audio_bank.items():
                if character_name not in actor_audio_bank[movie_title]:
                    continue
                if len(feature_list) < k:
                    feature_list.extend(random.sample(actor_audio_bank[movie_title][character_name], k - len(feature_list)))
        return audio_example_bank

    audio_example_bank = augment_audio_bank(audio_example_bank, actor_audio_bank)

    # Initialize audio feature extractor
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    predictions = {}
    for movie_title, per_movie_annotations in tqdm(annotations.items(), total=len(annotations)):
        
        # Find the video indices corresponding to the current movie title
        video_list = movie_video_list[movie_title]

        clip_idx_to_year = {}
        for video_idx in video_list:
            year = video_idx.split("/")[0]
            clip_idx = video_idx.split("/")[-1]
            clip_idx_to_year[clip_idx] = year

        # Load the audio feature bank for particular movies
        current_audio_example_bank = audio_example_bank[movie_title]
            
        per_movie_predictions = {}
        for clip_idx, per_clip_annotations in tqdm(per_movie_annotations.items(), total=len(per_movie_annotations)):

            # Path to the audio file
            year = clip_idx_to_year[clip_idx]
            audio_path = os.path.join(args.audio_dir, year, clip_idx + ".wav")

            per_clip_predictions = []

            if not args.cluster:
                for idx, segment in tqdm(enumerate(per_clip_annotations), total=len(per_clip_annotations)):
                    # Crop the speech segment and save to a temporary folder
                    start_time, end_time = segment["temporal_range"]
                    temporary_audio_path = os.path.join(args.temp_audio_dir, str(idx) + ".wav")
                    crop_audio(audio_path, temporary_audio_path, start_time, end_time)

                    # Extract the audio feature and match with the audio feature bank
                    signal, fs = torchaudio.load(temporary_audio_path)
                    query_feature = classifier.encode_batch(signal)[0][0]
                    query_feature = query_feature.detach().cpu().numpy()
                    pred = identify_active_speaker(query_feature, current_audio_example_bank)

                    # Write the predictions
                    segment["prediction"] = pred["identified_speaker"]
                    segment["score"] = pred["identified_score"]

                    per_clip_predictions.append(segment)
            else:
                # Gather features within a clip for clustering
                clip_all_features = []
                for idx, segment in tqdm(enumerate(per_clip_annotations), total=len(per_clip_annotations)):
                    # Crop the speech segment and save to a temporary folder
                    start_time, end_time = segment["temporal_range"]
                    temporary_audio_path = os.path.join(args.temp_audio_dir, str(idx) + ".wav")
                    crop_audio(audio_path, temporary_audio_path, start_time, end_time)

                    signal, fs = torchaudio.load(temporary_audio_path)
                    query_feature = classifier.encode_batch(signal)[0][0]
                    query_feature = query_feature.detach().cpu().numpy()
                    
                    clip_all_features.append(query_feature)

                # Cluster features and get the centroids
                clip_clusters, cluster_centroids = group_speakers(clip_all_features)

                # Identify each cluster
                speaker_idx_to_label = {}
                for speaker_idx in clip_clusters:
                    cluster_centroid = cluster_centroids[speaker_idx]
                    results = identify_active_speaker(cluster_centroid, current_audio_example_bank)

                    speaker_idx_to_label[speaker_idx] = {"label": results["identified_speaker"], "score": results["identified_score"]}

                # Write the prediction results for the cluster
                for idx, segment in enumerate(per_clip_annotations):
                    start_time, end_time = segment["temporal_range"]
                    segment_feature = clip_all_features[idx].reshape(1, -1)

                    query_speaker_idx = None
                    for speaker_idx, index_list in clip_clusters.items():
                        if idx in index_list:
                            query_speaker_idx = speaker_idx

                    segment["prediction"] = speaker_idx_to_label[query_speaker_idx]["label"]
                    # Scale the confidence score by the distance to the centroid of the cluster
                    segment["score"] = float(speaker_idx_to_label[query_speaker_idx]["score"] * cosine_similarity(segment_feature, cluster_centroids[query_speaker_idx]))

                    per_clip_predictions.append(segment)

        predictions[movie_title] = per_movie_predictions

    # Save the voice identification results to a JSON file
    with open(args.save_predictions_file, 'w') as outfile:
        json.dump(predictions, outfile, indent=4)

    # Empty the temporary folder
    shutil.rmtree(args.temp_audio_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_annotation_file", type=str, default=None, help="Path to the corrected character bank audio annotation JSON file")
    parser.add_argument("--movie_to_video_file", type=str, default=None, help="Path to the movie-to-video mapping JSON file")
    parser.add_argument("--example_audio_file", type=str, default=None, help="Path to the example audio bank JSON file")
    parser.add_argument("--actor_audio_bank_file", type=str, default=None, help="Path to the actor audio bank JSON file")
    parser.add_argument("--audio_dir", type=str, default=None, help="Directory containing animated audio files")
    parser.add_argument("--temp_audio_dir", type=str, default=None, help="Temporary directory for actor audio processing")
    parser.add_argument("--save_predictions_file", type=str, default=None, help="Temporary directory for actor audio processing")
    parser.add_argument("--cluster", action="store_true")
    args = parser.parse_args()

    main()
