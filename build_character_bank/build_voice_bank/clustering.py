import argparse
import json
import os
import pandas as pd
import subprocess

import numpy as np
import torch
import torchaudio

from glob import glob
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import whisperx

from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


def load_json(file_path):
    with open(file_path, 'r') as infile:
        return json.load(infile)


def crop_audio(input_video, output_video, start_time, end_time):
    if os.path.exists(output_video):
        return
    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", input_video,
        "-c:v", "copy",
        "-ar", "16000",
        "-ac", "1",
        output_video
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return
    

def average_normalized_embeddings(embeddings_list):
    """Find the centroid of a group of features."""
    normalized_embeds = []
    for emb in embeddings_list:
        norm = np.linalg.norm(emb)
        if norm != 0:
            normalized_embeds.append(emb / norm)
        else:
            normalized_embeds.append(emb)
    avg_embed = np.mean(normalized_embeds, axis=0)
    final_norm = np.linalg.norm(avg_embed)
    if final_norm != 0:
        return avg_embed / final_norm
    return avg_embed


def group_global_speakers(flattened_data, threshold=0.25):
    """Cluster the speakers across all files in a greedy and online way."""
    clusters = []
    for f_idx, spk_id, emb in flattened_data:
        emb_2d = emb.reshape(1, -1)
        placed = False
        for i, (rep_embs, spk_entries) in enumerate(clusters):
            sim_vals = cosine_similarity(emb_2d, rep_embs)
            if np.any(sim_vals > threshold):
                spk_entries.append((f_idx, spk_id))
                clusters[i] = (np.vstack([rep_embs, emb_2d]), spk_entries)
                placed = True
                break
        if not placed:
            clusters.append((emb_2d, [(f_idx, spk_id)]))
    return [c[1] for c in clusters]


## Below is an alternative for clustering speakers, which is not dependent on the order of the input sequence. 
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics.pairwise import cosine_distances

# def group_global_speakers(flattened_data, threshold=0.25):
#     embeddings = np.array([emb for _, _, emb in flattened_data])
#     speaker_info = [(f_idx, spk_id) for f_idx, spk_id, _ in flattened_data]

#     # Compute cosine *distances* (not similarities)
#     distances = cosine_distances(embeddings)
    
#     # Convert similarity threshold to distance threshold
#     distance_threshold = 1 - threshold

#     clustering = AgglomerativeClustering(
#         n_clusters=None,  # Let the threshold decide
#         affinity='precomputed',
#         linkage='average',
#         distance_threshold=distance_threshold
#     )
#     labels = clustering.fit_predict(distances)

#     # Group speaker_info by cluster labels
#     clusters = {}
#     for label, info in zip(labels, speaker_info):
#         clusters.setdefault(label, []).append(info)
#     return list(clusters.values())


def match_speakers(actor_features):
    file_to_speakers = {}
    for file_idx, spk_dict in actor_features.items():
        file_to_speakers[file_idx] = {}
        for local_spk_id, embeddings_list in spk_dict.items():
            file_to_speakers[file_idx][local_spk_id] = average_normalized_embeddings(embeddings_list)
    flattened_data = []
    for file_idx, spk_dict in file_to_speakers.items():
        for local_spk_id, avg_emb in spk_dict.items():
            flattened_data.append((file_idx, local_spk_id, avg_emb))
    clusters = group_global_speakers(flattened_data)
    return clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', default=None, type=str)
    parser.add_argument('--save_transcription_dir', default=None, type=str)
    parser.add_argument('--cast_information_file', default=None, type=str)
    parser.add_argument('--audio_clip_dir', default=None, type=str)
    parser.add_argument('--audio_feature_dir', default=None, type=str)
    parser.add_argument('--save_file', default=None, type=str)
    args = parser.parse_args()

    src_dir = args.src_dir
    save_transcription_dir = args.save_transcription_dir
    cast_information_file = args.cast_information_file
    audio_clip_dir = args.audio_clip_dir
    audio_feature_dir = args.audio_feature_dir
    save_file = args.save_file

    # Transcribe interview videos to text using WhisperX
    for audio_file in tqdm(glob(os.path.join(src_dir, "*", "*", "*"))):
        movie_title = audio_file.split("/")[-3]
        actor_name = audio_file.split("/")[-2]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 16 # reduce if low on GPU mem
        compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size, language="en")

        # 2. Align whisper output
        # prepare Align model
        model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
        result_align = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # 3. Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(use_auth_token="HF_TOKEN", device=device)
        diarize_segments = diarize_model(audio)
        result_diarize = whisperx.assign_word_speakers(diarize_segments, result_align)

        # 4. Save the diarized transcription
        starts = []
        ends = []
        texts = []
        speakers = []
        
        for result_diarize_single in result_diarize['segments']:
            if 'speaker' in result_diarize_single.keys():
                speakers.append(result_diarize_single['speaker'])
            else:
                speakers.append("Unknown")
            starts.append(result_diarize_single['start'])
            ends.append(result_diarize_single['end'])
            texts.append(result_diarize_single['text'])
            
        save_path = audio_file.replace(src_dir, save_transcription_dir).replace(".wav", ".csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        output_df = pd.DataFrame.from_records({'start': starts, 'end': ends, 'text': texts, 'speaker': speakers})
        output_df.to_csv(save_path)

    # Cluster the speakers in the transcribed interviews
    cast_information = load_json(cast_information_file)

    feature_bank = {}
    for movie_title, movie_cast in cast_information.items():
        actor_name_to_character_name = {}
        for character_name, actor_name in movie_cast.items():
            actor_name_to_character_name[actor_name] = character_name

        movie_feature_bank = {}
        for actor_name in tqdm(os.listdir(os.path.join(save_transcription_dir, movie_title)), total=len(os.listdir(os.path.join(save_transcription_dir, movie_title)))):
            try:
                character_name = actor_name_to_character_name[actor_name]

                actor_features = {}
                feature_pt = {}
                file_list = glob(os.path.join(save_transcription_dir, movie_title, actor_name, "*"))

                for file in tqdm(file_list, total=len(file_list)):
                    df = pd.read_csv(file)
                    file_idx = file.split("/")[-1][:-4]
                    audio_file = file.replace(save_transcription_dir, src_dir).replace(".csv", ".wav")

                    # Extract time ranges and speaker indices
                    time_range_list = []
                    speaker_idx_list = []
                    for _, row in df.iterrows():
                        start_time = row["start"]
                        end_time = row["end"]

                        ## Optional, discard segments if too short
                        # if end_time - start_time < 3:
                        #     continue

                        time_range_list.append((start_time, end_time))
                        speaker_idx_list.append(row["speaker"])

                    clip_features = []
                    clip_feature_path_list = []
                    for idx, time_range in tqdm(enumerate(time_range_list), total=len(time_range_list)):
                        try:
                            # Crop the audio segment and extract the audio feature
                            output_audio = os.path.join(audio_clip_dir, movie_title, actor_name, f"{file_idx}_{str(idx)}.wav")
                            save_filename = output_audio.replace(".wav", ".npz").replace(audio_clip_dir, audio_feature_dir)

                            if os.path.exists(save_filename):
                                query_feature = np.load(save_filename)['feature']
                            else:
                                os.makedirs(os.path.dirname(output_audio), exist_ok=True)
                                crop_audio(audio_file, output_audio, time_range[0], time_range[1])

                                signal, fs = torchaudio.load(output_audio)
                                query_feature = classifier.encode_batch(signal)[0][0]
                                query_feature = query_feature.detach().cpu().numpy()

                                os.makedirs(os.path.dirname(save_filename), exist_ok=True)
                                np.savez(save_filename, feature=query_feature)

                            clip_features.append(query_feature)
                            clip_feature_path_list.append(save_filename)
                        except:
                            continue

                    # Group features by speaker
                    clip_speakers = {}
                    feature_pt[file_idx] = {}
                    for speaker_idx, feature, feature_path in zip(speaker_idx_list, clip_features, clip_feature_path_list):
                        if speaker_idx not in clip_speakers:
                            clip_speakers[speaker_idx] = [feature]
                        else:
                            clip_speakers[speaker_idx].append(feature)

                        if speaker_idx not in feature_pt[file_idx]:
                            feature_pt[file_idx][speaker_idx] = [feature_path]
                        else:
                            feature_pt[file_idx][speaker_idx].append(feature_path)

                    actor_features[file_idx] = clip_speakers

                # Match speakers using clustering across all files
                clusters = match_speakers(actor_features)
                
                # Identify the largest cluster
                max_len = 0
                max_idx = None
                for idx, speaker_cluster in enumerate(clusters):
                    if len(speaker_cluster) >= max_len:
                        max_len = len(speaker_cluster)
                        max_idx = idx

                identified_speaker_cluster = clusters[max_idx]

                # Collect paths to feature (.npz) files for each actor
                actor_examples = []
                for (file_idx_, speaker_idx_) in identified_speaker_cluster:
                    actor_examples.extend(feature_pt[file_idx_][speaker_idx_])

                movie_feature_bank[character_name] = actor_examples
                
            except:
                continue

        feature_bank[movie_title] = movie_feature_bank

    # Save the mapping from actors to feature paths into a .json file
    with open(save_file, 'w') as outfile:
        json.dump(feature_bank, outfile, indent=4)
    