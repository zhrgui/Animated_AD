import os
import sys
import ast
import json
import ipdb.stdout
import torch
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

import numpy as np
import pandas as pd
from PIL import Image
from qwen_vl_utils import smart_resize
from utils import process_video, expand2square, convert_bounding_box_to_rectangle, convert_bounding_box_to_ellipse, text_to_token

import ipdb

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))

VIS_DIR = None

def fetch_video_from_tensor(video, image_factor: int = IMAGE_FACTOR):
    nframes, _, height, width = video.shape
    ele = {}
    ele["max_pixels"] = 360 * 420
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels = ele.get("max_pixels", max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video

class CMDAD_Animated_FrameLoader():
    def __init__(self,
            tokenizer,
            processor,
            general_prompt,
            video_type, 
            label_type, 
            label_width, 
            label_alpha,
            anno_path,
            video_dir,
            image_mode,
            **kwargs):
        self.processor = processor
        self.tokenizer = tokenizer
        self.general_prompt=general_prompt
        self.video_type = video_type

        # label information, including colour coding, type, etc.
        self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.color_name = ["red", "green", "blue", "yellow", "pink", "cyan", "white", "black", "black", "black", "black", "black"]
        self.label_type=label_type
        self.label_width=label_width
        self.label_alpha=label_alpha

        # load annotation file
        self.anno_df = pd.read_csv(anno_path)
        self.anno_df['num_words'] = self.anno_df.apply(lambda x: len(x['text'].strip().split()), axis=1)
        self.anno_df = self.anno_df[(self.anno_df['num_words'] < 64) & (self.anno_df['num_words'] > 1)]
        self.anno_df['num_string'] = self.anno_df.apply(lambda x: len(x['text']), axis=1)
        self.anno_df = self.anno_df[self.anno_df['num_string'] < 250]

        self.image_mode = image_mode

        self.all_clips = []
        for anno_idx, anno_row in self.anno_df.iterrows():
            cmd_filename = anno_row['cmd_filename']
            if "anno_idx" in anno_row.keys():
                anno_idx = int(anno_row["anno_idx"])
            video_path = os.path.join(video_dir, cmd_filename + '.mkv')
            if os.path.exists(video_path):
                self.all_clips.append((anno_row["imdbid"], video_path, anno_idx, anno_row["scaled_start"], anno_row["scaled_end"], anno_row["audiovault_start"], anno_row["audiovault_end"], anno_row["text"], anno_row["bboxes"], anno_row["pred_ids"]))

        print(f"In total {len(self.all_clips)} CMD-AD clips")

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, index):
        imdbid, video_path, anno_idx, start, end, start_, end_, gt_text, bboxes, name_ids = self.all_clips[index]
        
        frames, _ = process_video(video_path, num_frames=8, start=start, end=end)
        
        bboxes = ast.literal_eval(bboxes)
        bboxes = bboxes[::2]
        name_ids = ast.literal_eval(name_ids)
        bboxes_filtered = []
        all_name_ids = {}
        
        if self.label_type != "none":
            for frame_idx in range(len(frames)):
                bboxes_filtered_per_frame = {}
                for name_idx, (name_id, bbox_idx_list) in enumerate(name_ids.items()):
                    for bbox_idx in bbox_idx_list:
                        if bbox_idx[0] == int(frame_idx * 2) and name_id not in bboxes_filtered_per_frame.keys():
                            bboxes_filtered_per_frame[name_id] = bboxes[frame_idx][bbox_idx[1]]
                            if name_id not in all_name_ids.keys():
                                all_name_ids[name_id] = len(all_name_ids)
                bboxes_filtered.append(bboxes_filtered_per_frame)

        if self.label_type=="none":
            # processed_frames = [expand2square(frame, tuple(int(x*255) for x in self.processor_mean)) for frame in frames]
            processed_frames = []
            for frame_idx, frame in enumerate(frames):
                processed_frames.append(frame)
        else:
            processed_frames = []
            for frame_idx, frame in enumerate(frames):
                if len(bboxes_filtered[frame_idx]) == 0: # skip as no bbox presents in this frame
                    if self.image_mode == "padding":
                        processed_frames.append(expand2square(frame, tuple(int(x*255) for x in self.processor_mean)))
                    elif self.image_mode == "default":
                        processed_frames.append(frame)
                    continue
                else:
                    label_masks = None
                    total_masks = None
                    for b_idx, (name_id, bbox) in enumerate(bboxes_filtered[frame_idx].items()):
                        # draw binary label masks
                        if self.label_type=="boxes":
                            label_mask = convert_bounding_box_to_rectangle(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        elif self.label_type=="circles":
                            label_mask = convert_bounding_box_to_ellipse(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        else:
                            print("Check the label type")
                            sys.exit()

                        # overlay label masks to get an overall mask
                        if label_masks is None:
                            label_masks = label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = label_mask
                        else:
                            label_masks = label_masks * (1 - label_mask[:, :, None]) + label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = np.clip(label_mask + total_masks, 0., 1.)
                    # overlay the overall label mask onto the frame
                    processed_frame = Image.fromarray((np.array(frame) * (1- total_masks[:, :, None] * self.label_alpha) + total_masks[:, :, None] * self.label_alpha * label_masks).astype(np.uint8))
                    if self.image_mode == "padding":
                        processed_frame = expand2square(processed_frame, tuple(int(x*255) for x in self.processor_mean)) 
                    processed_frames.append(processed_frame)

        processed_frames = np.stack(processed_frames, 0)
        video_tensor = torch.from_numpy(processed_frames).permute(0, 3, 1, 2)

        char_text = ". Possible characters (labeled by {label_type}): "
        for name_idx, (name_id, color_idx) in enumerate(all_name_ids.items()):
            if name_idx == len(all_name_ids) - 1:
                ending = ""
            else:
                ending = ", "
            char_text = char_text + name_id + " (" + self.color_name[color_idx] + ")" + ending
        
        if char_text == ". Possible characters (labeled by {label_type}): ": # no character recognised
            prompt = self.general_prompt.format(video_type=self.video_type, char_text="", label_type=self.label_type, duration=str(round(end_-start_, 2)))
            input_id, text_prompt = text_to_token(prompt, self.processor)
        else:
            prompt = self.general_prompt.format(video_type=self.video_type, char_text=char_text.format(label_type=self.label_type), label_type=self.label_type, duration=str(round(end_-start_, 2)))
            input_id, text_prompt = text_to_token(prompt, self.processor)

        return_dict =  {'video': video_tensor,
                'imdbid': imdbid,
                'input_id': input_id,
                'prompt': text_prompt,
                'gt_text': gt_text,
                'start': start, 
                'end': end, 
                'start_': start_, 
                'end_': end_,
                'anno_idx': anno_idx,
                }
        return return_dict

    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['imdbid'] = [sample['imdbid'] for sample in batch]
        out_batch['video'] = [sample['video'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['start_'] = [sample['start_'] for sample in batch]
        out_batch['end_'] = [sample['end_'] for sample in batch]
        out_batch['input_id'] = [sample['input_id'] for sample in batch]
        out_batch['prompt'] = [sample['prompt'] for sample in batch]
        out_batch['gt_text'] = [sample['gt_text'] for sample in batch]
        out_batch['anno_idx'] = [sample['anno_idx'] for sample in batch]
        return out_batch

