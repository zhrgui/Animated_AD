import os
import numpy as np
import pandas as pd

import torch
import torchaudio

import random

import subprocess
from glob import glob
from tqdm import tqdm

from speechbrain.inference.speaker import EncoderClassifier

import json

with open("/scratch/shared/beegfs/zhongrui/autoad_data/annotations/movie_to_video_files.json", 'r') as infile:
    movie_to_video = json.load(infile)

clip_idx_to_movie = {}
for movie, videos in movie_to_video.items():
    for video in videos:
        clip_idx = video.split("/")[-1]
        clip_idx_to_movie[clip_idx] = movie

CMD_SUBSET_MOVIES = [
    "Coraline", "Despicable Me", "Ice Age", "Kung Fu Panda", 
    "Spider Man: Into the Spider-Verse", "Sing", 
    "The Secret Life of Pets", "The Polar Express"
]

def crop_audio(input_video, output_video, start_time, end_time):
    if os.path.exists(output_video):
        return
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
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")
    dot_product = np.dot(arr1, arr2)
    magnitude_arr1 = np.linalg.norm(arr1)
    magnitude_arr2 = np.linalg.norm(arr2)

    if magnitude_arr1 == 0 or magnitude_arr2 == 0:
        return 0.0
    cosine_sim = dot_product / (magnitude_arr1 * magnitude_arr2)
    return cosine_sim

def verify_features(feature_a, feature_b):
    score = cosine_similarity(feature_a, feature_b)
    prediction = score >= 0.25
    return score, prediction

def load_audio_feature_bank(example_audio_file):
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

# def identify_active_speaker(query_feature, audio_example_bank):
#     char_to_score = {}

#     for char_name, feature_list in audio_example_bank.items():
#         classification_score = 0.

#         for feature in feature_list:
#             score, prediction = verify_features(query_feature, feature)
#             classification_score += float(prediction)
        
#         classification_score = classification_score / len(feature_list)
#         char_to_score[char_name] = classification_score

#     identified_char_name = max(char_to_score, key=char_to_score.get)
#     return identified_char_name, char_to_score[identified_char_name]

def identify_active_speaker(query_feature, audio_example_bank):
    char_to_score = {}

    for char_name, feature_list in audio_example_bank.items():
        classification_score = 0.

        cosine_sim = []
        for feature in feature_list:
            score, prediction = verify_features(query_feature, feature)
            cosine_sim.append(score)

        cosine_sim.sort(reverse=True)

        if len(cosine_sim) <= 3:
            classification_score = sum(cosine_sim) / len(cosine_sim)
        else:
            classification_score = sum(cosine_sim[:3]) / 3
        
        # classification_score = classification_score / len(feature_list)
        char_to_score[char_name] = classification_score

    identified_char_name = max(char_to_score, key=char_to_score.get)
    return identified_char_name, char_to_score[identified_char_name]

if __name__ == "__main__":
    transcription_dir = "/scratch/shared/beegfs/zhongrui/animated_transcripts"
    audio_dir = "/scratch/shared/beegfs/zhongrui/animated_audios"
    example_audio_file = "/work/zhongrui/avobjects_zgui/audio_example_bank.json"
    temp_audio_dir = "/scratch/shared/beegfs/zhongrui/temp_aud"
    save_dir = "/scratch/shared/beegfs/zhongrui/animated_transcripts_character_no_expansion"
    threshold = 0.45

    actor_audio_bank_file = "/scratch/shared/beegfs/zhongrui/cmd_character_bank/cmd_actor_examples.json"

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    audio_example_bank = load_audio_feature_bank(example_audio_file)
    actor_audio_bank = load_audio_feature_bank(actor_audio_bank_file)


    def augment_audio_bank(audio_example_bank, actor_audio_bank, k = 15):
        for movie_title, per_movie_audio_bank in audio_example_bank.items():
            for character_name, feature_list in per_movie_audio_bank.items():
                if character_name not in actor_audio_bank[movie_title]:
                    continue
                if len(feature_list) < k:
                    feature_list.extend(random.sample(actor_audio_bank[movie_title][character_name], k - len(feature_list)))
        return audio_example_bank
    

    audio_example_bank = augment_audio_bank(audio_example_bank, actor_audio_bank)


    transcript_file_list = glob(os.path.join(transcription_dir, "*", "*"))
    for transcript_file in transcript_file_list:
        anno_df = pd.read_csv(transcript_file)
        anno_df_with_char = pd.DataFrame(columns=anno_df.columns)

        anno_df_with_char = anno_df_with_char.drop(columns=['speaker'])

        clip_idx = os.path.basename(transcript_file).split(".")[0]
        year = transcript_file.split("/")[-2]
        movie_title = clip_idx_to_movie[clip_idx]

        if movie_title not in CMD_SUBSET_MOVIES:
            continue

        current_audio_example_bank = audio_example_bank[movie_title]

        # import ipdb; ipdb.set_trace()

        for row_idx, row in tqdm(anno_df.iterrows(), total=len(anno_df)):
            start_time = row["start"]
            end_time = row["end"]

            audio_path = os.path.join(audio_dir, year, clip_idx + ".wav")
            temporary_audio_path = os.path.join(temp_audio_dir, str(row_idx) + ".wav")

            crop_audio(audio_path, temporary_audio_path, start_time, end_time)

            signal, fs = torchaudio.load(temporary_audio_path)
            query_feature = classifier.encode_batch(signal)[0][0]
            query_feature = query_feature.detach().cpu().numpy()
            identified_character, score = identify_active_speaker(query_feature, current_audio_example_bank)

            expand_feature_bank = False
            if score <= threshold:
                # identified_character = "Uncertain"
                pass
            else:
                if expand_feature_bank:
                    current_audio_example_bank[identified_character].append(query_feature)

            del row["speaker"]
            row["character"] = identified_character
            row["score"] = score
            # import ipdb; ipdb.set_trace()
            anno_df_with_char = pd.concat([anno_df_with_char, pd.DataFrame([row])], ignore_index=True)

        save_path = os.path.join(save_dir, movie_title, transcript_file.split("/")[-1])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        anno_df_with_char.to_csv(save_path, index=False)
