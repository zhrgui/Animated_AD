import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from torch.utils.data import DataLoader, Subset
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
assert "VIDEOLLAMA2_PATH" in os.environ, "Set VIDEOLLAMA2_PATH environment variable to point to your VideoLLaMA2 installation"
sys.path.append(os.environ["VIDEOLLAMA2_PATH"])
from videollama2.model.builder import load_pretrained_model

from promptloader import get_general_prompt
from dataloader import MADEval_FrameLoader, TVAD_FrameLoader, CMDAD_FrameLoader, CMDAD_Animated_FrameLoader


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
    # initialize VideoLLaMA2
    model_path = args.model_path
    model_name = "VideoLLaMA2-7B" 
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.cuda()

    # formulate text prompt template
    general_prompt = get_general_prompt(args.prompt_idx)
    
    # build dataloader
    if args.dataset == "tvad":
        D = TVAD_FrameLoader
        video_type = "TV series"
    elif args.dataset == "cmdad":
        D = CMDAD_FrameLoader
        # D = CMDAD_Animated_FrameLoader
        video_type = "movie"
    elif args.dataset == "madeval":
        D = MADEval_FrameLoader
        video_type = "movie"
    else:
        print("Check dataset name")
        sys.exit()

    # ad_dataset = D(tokenizer=tokenizer, processor=processor, general_prompt=general_prompt, video_type = video_type,
    #                             anno_path=args.anno_path, video_dir=args.video_dir,
    #                             label_type=args.label_type, label_width=args.label_width, label_alpha=args.label_alpha)
    
    ad_dataset = D(tokenizer=tokenizer, processor=processor, general_prompt=general_prompt, video_type = video_type,
                                anno_path=args.anno_path, video_dir=args.video_dir,
                                label_type=args.label_type, label_width=args.label_width, label_alpha=args.label_alpha, charbank_path=args.charbank_path)
              
    loader = DataLoader(ad_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                            collate_fn=ad_dataset.collate_fn, shuffle=False, pin_memory=True)

    start_sec = []
    end_sec = []
    start_sec_ = []
    end_sec_ = []
    text_gt = []
    text_gen = []
    vids = []

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
        videos = [video.cuda() for video in input_data["video"]]
        modal_list = ['video'] * len(videos)
        input_ids = input_data["input_id"]

        # left padding for batch inference
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x.flip(dims=[0]) for x in input_ids],
            batch_first=True,
            padding_value=tokenizer.pad_token_id).flip(dims=[1]).cuda()
        attention_mask=input_ids.ne(tokenizer.pad_token_id).cuda()

        # import ipdb; ipdb.set_trace()
        
        # repeat experiments max_exp
        redo_exp = True
        counter = 0
        while counter < args.max_exp and redo_exp:
            # if counter >= 1:
            #     print("Redoing experiment due to over-length (unstable) outputs")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images_or_videos=videos,
                    modal_list=modal_list,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [output.replace("\n", " ") for output in outputs]
            redo_exp = False
            counter += 1
            for output in outputs:
                if len(output) > 2000: # if output length > 2000, assume it is unstable and redo the experiment
                    redo_exp = True

        # import ipdb; ipdb.set_trace()
        vids.extend(input_data["imdbid"])
        text_gt.extend(input_data["gt_text"])
        text_gen.extend(outputs) 
        start_sec.extend(input_data["start"])
        end_sec.extend(input_data["end"])
        start_sec_.extend(input_data["start_"])
        end_sec_.extend(input_data["end_"])
        
    output_df = pd.DataFrame.from_records({'vid': vids, 'start': start_sec, 'end': end_sec, 'start_': start_sec_, 'end_': end_sec_, 'text_gt': text_gt, 'text_gen': text_gen})
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    print(f"Save csv to {args.output_dir}")
    output_df.to_csv(f'{args.output_dir}/stage1_{args.chunk_idx}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('llama')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dataset', default="cmdad", type=str)
    parser.add_argument('--prompt_idx', default=0, type=int)
    parser.add_argument('--model_path', required=True, type=str, help='Path to VideoLLaMA2 model')
    parser.add_argument('--anno_path', required=True, type=str, help='Path to annotation CSV')
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--video_dir', required=True, type=str, help='Path to video directory')
    parser.add_argument('--charbank_path', required=True, type=str, help='Path to character bank JSON')
    parser.add_argument('--label_type', default="boxes", type=str)
    parser.add_argument('--label_width', default=10, type=int, help='label_width, 10 in a canvas 1000')
    parser.add_argument('--label_alpha', default=0.8, type=float)
    parser.add_argument('-j', '--num_workers', default=8, type=int, help='init mode')
    parser.add_argument('--max_exp', default=5, type=int, help='maximum number of repeating experiments')
    
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--chunk_idx", type=int, default=0)

    args = parser.parse_args()

    main(args)

    
    