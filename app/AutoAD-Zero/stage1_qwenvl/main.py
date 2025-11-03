import os
import torch
from torch.utils.data import DataLoader, Subset

import argparse
import pandas as pd
from tqdm import tqdm

from promptloader import get_general_prompt
from dataloader import CMDAD_Animated_FrameLoader
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    if n <= 0:
        raise ValueError("Number of chunks must be positive")
    if not lst:
        return [[] for _ in range(n)]

    q, r = divmod(len(lst), n)
    chunks = []
    start = 0

    for i in range(n):
        end = start + q + (1 if i < r else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def main(args):
    if args.model == "qwen2_vl":
        if args.model_type == "7B":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
            )
        elif args.model_type == "72B-AWQ":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-72B-Instruct-AWQ", torch_dtype=torch.float16, device_map="auto"
            )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    elif args.model == "qwen2_5_vl":
        if args.model_type == "7B":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
            )
        elif args.model_type == "72B-AWQ":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", torch_dtype=torch.float16, device_map="auto"
            )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    general_prompt = get_general_prompt(args.prompt_idx)

    ad_dataset = CMDAD_Animated_FrameLoader(tokenizer=None, processor=processor, general_prompt=general_prompt, video_type = "movie",
                                            anno_path=args.anno_path, video_dir=args.video_dir, label_type=args.label_type, 
                                            label_width=args.label_width, label_alpha=args.label_alpha, image_mode=args.image_mode)
    
    loader = DataLoader(ad_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                        collate_fn=ad_dataset.collate_fn, shuffle=False, pin_memory=True)

    start_sec = []
    end_sec = []
    start_sec_ = []
    end_sec_ = []
    text_gt = []
    text_gen = []
    text_gen_all = []
    vids = []
    anno_indices = []

    all_indices = list(range(len(ad_dataset)))
    chunks = split_list(all_indices, args.num_chunks)

    chunked_loaders = []
    for chunk_indices in chunks:
        subset = Subset(ad_dataset, chunk_indices)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=ad_dataset.collate_fn,
            shuffle=False,
            pin_memory=True
        )
        chunked_loaders.append(loader)

    for idx, input_data in tqdm(enumerate(chunked_loaders[args.chunk_idx]), total=len(chunked_loaders[args.chunk_idx]), desc='EVAL'): 
        video_inputs = input_data["video"]
        texts = input_data["input_id"]
        if args.model == "qwen2_vl":
            inputs = processor(
                text=texts,
                images=None,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        elif args.model == "qwen2_5_vl":
            fps = int(max(0.5, 16 / (input_data["end"][0] - input_data["start"][0])))
            inputs = processor(
                text=texts,
                images=None,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                fps = [fps],
            )
        inputs = inputs.to("cuda")

        redo_exp = True
        counter = 0
        while counter < args.max_exp and redo_exp:
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            redo_exp = False

            outputs_partitioned = [outputs[i * args.num_return_sequences : (i + 1) * args.num_return_sequences] for i in range(args.batch_size)]
            outputs_partitioned_0 = [outputs[i * args.batch_size : (i + 1) * args.batch_size] for i in range(args.num_return_sequences)][0]

            counter += 1
            for output in outputs_partitioned_0:
                if len(output) > 1500: # if output length > 2000, assume it is unstable and redo the experiment
                    redo_exp = True
                        
        anno_indices.extend(input_data["anno_idx"])
        vids.extend(input_data["imdbid"])
        text_gt.extend(input_data["gt_text"])
        text_gen.extend(outputs_partitioned_0) 
        text_gen_all.extend(outputs_partitioned)
        start_sec.extend(input_data["start"])
        end_sec.extend(input_data["end"])
        start_sec_.extend(input_data["start_"])
        end_sec_.extend(input_data["end_"])

    output_df = pd.DataFrame.from_records({'anno_idx': anno_indices, 'vid': vids, 'start': start_sec, 'end': end_sec, 'start_': start_sec_, 'end_': end_sec_, 'text_gt': text_gt, 'text_gen': text_gen})
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    print(f"Save csv to {args.output_dir}")
    output_df.to_csv(f'{args.output_dir}/stage1_{args.chunk_idx}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('llama')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--dataset', default="cmdad", type=str)
    parser.add_argument('--num_return_sequences', default=1, type=int)
    parser.add_argument('--prompt_idx', default=0, type=int)
    parser.add_argument('--model', default="qwen2_vl", type=str)
    parser.add_argument('--model_type', default="7B", type=str)
    parser.add_argument('--image_mode', default="default", type=str)
    parser.add_argument('--sampling_mode', default="multinomial", type=str)
    parser.add_argument('--frame_sampling', default=None, type=str)
    parser.add_argument('--anno_path', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--video_dir', default="/scratch/shared/beegfs/zhongrui/animated_videos", type=str)
    parser.add_argument('--label_type', default="boxes", type=str)
    parser.add_argument('--label_width', default=10, type=int, help='label_width, 10 in a canvas 1000')
    parser.add_argument('--label_alpha', default=0.8, type=float)
    parser.add_argument('-j', '--num_workers', default=8, type=int, help='init mode')
    parser.add_argument('--max_exp', default=1, type=int, help='maximum number of repeating experiments')
    parser.add_argument('--num_frames', default=8, type=int, help='number of frames')
    parser.add_argument('--dev', action='store_true')

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)

    args = parser.parse_args()

    main(args)
