import argparse
import os

os.environ['NUMBA_CACHE_DIR'] = ""
# os.environ["WANDB_MODE"] = "disabled"

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data as data_utils
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
import wandb

from lwtnet import *
from dataset import *
from load_audio import wav2filterbanks

from tqdm import tqdm
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


cross_entropy_loss = nn.CrossEntropyLoss()
cos_sim = nn.CosineSimilarity(dim=1)
scaler = torch.amp.GradScaler()


def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cosine_loss_attn(a, v, model, args, device):
    with torch.amp.autocast():
        time_size = a.size(2)
        losses_row = []
        losses_col = []
        p1s_row, p5s_row = [], []
        p1s_col, p5s_col = [], []
        label = torch.arange(time_size).to(device)
        for i in range(a.size(0)):
            ft_v = v[[i], :, :].transpose(2, 0)
            ft_a = a[[i], :, :].transpose(2, 0)

            spatial_sim = cos_sim(ft_v.expand(-1, -1, time_size, -1, -1),
                                ft_a.expand(-1, -1, time_size).transpose(0, 2).unsqueeze(-1).unsqueeze(-1))

            if args.similarity_type == "mean":
                mean_sim = torch.mean(spatial_sim.reshape(spatial_sim.shape[0], spatial_sim.shape[1], -1), axis=-1)
                output = model.logits_scale(mean_sim.unsqueeze(-1)).squeeze(-1)
            elif args.similarity_type == "max":
                max_sim = torch.amax(spatial_sim, dim=(2, 3))
                output = model.logits_scale(max_sim.unsqueeze(-1)).squeeze(-1)

                mean_sim = torch.mean(spatial_sim.reshape(spatial_sim.shape[0], spatial_sim.shape[1], -1), axis=-1)
                # output_mean = model.logits_scale(mean_sim.unsqueeze(-1)).squeeze(-1)

    # target_mean = torch.zeros_like(output_mean).to(output_mean.device)
    target_mean = torch.zeros_like(mean_sim).to(mean_sim.device)
    if args.scale_mean:
        mean_sim = (mean_sim + 1) / 2
    bce_loss = F.binary_cross_entropy(mean_sim, target_mean)

    losses_row.append(cross_entropy_loss(output, label))
    p1, p5 = accuracy(output, label)
    p1s_row.append(p1.item())
    p5s_row.append(p5.item())

    losses_col.append(cross_entropy_loss(output.T, label))
    p1, p5 = accuracy(output.T, label)
    p1s_col.append(p1.item())
    p5s_col.append(p5.item())

    loss_row = sum(losses_row) / len(losses_row)
    p1_row = sum(p1s_row) / len(p1s_row)
    p5_row = sum(p5s_row) / len(p5s_row)

    loss_col = sum(losses_col) / len(losses_col)
    p1_col = sum(p1s_col) / len(p1s_col)
    p5_col = sum(p5s_col) / len(p5s_col)

    return loss_row, p1_row, p5_row, loss_col, p1_col, p5_col, bce_loss


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def main(rank, world_size, args):
    setup(rank, world_size)

    hparams = OmegaConf.load(args.config)

    device = torch.device(f"cuda:{rank}")
    hparams.syncnet_batch_size = hparams.syncnet_batch_size // world_size

    nepochs = hparams.nepochs
    checkpoint_dir = args.checkpoint_dir
    checkpoint_interval = hparams.syncnet_checkpoint_interval

    if rank == 0:
        wandb.login()
        wandb.init(project="avobjects_finetune", entity="g2zr004", config=args)

    model = LWTNet_temporal_adapter().to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hparams.start_syncnet_lr)

    global_epoch = 0
    global_step = 0

    if args.synthetic:
        train_dataset = DataGenerator_Synthetic(hparams, height=args.height, width=args.width, fps=args.fps, sample_rate=args.sample_rate, test=False)
        test_dataset = DataGenerator_Synthetic(hparams, height=args.height, width=args.width, fps=args.fps, sample_rate=args.sample_rate, test=True)
    else:
        train_dataset = DataGenerator_Fulldata(hparams, height=args.height, width=args.width, fps=args.fps, sample_rate=args.sample_rate, test=False)
        test_dataset = DataGenerator_Fulldata(hparams, height=args.height, width=args.width, fps=args.fps, sample_rate=args.sample_rate, test=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_data_loader = data_utils.DataLoader(train_dataset, batch_size=hparams.syncnet_batch_size, sampler=train_sampler, num_workers=hparams.num_workers)
    test_data_loader = data_utils.DataLoader(test_dataset, batch_size=hparams.syncnet_batch_size, sampler=test_sampler, num_workers=hparams.num_workers)

    num_training_steps = nepochs * len(train_data_loader)
    num_warmup_steps = hparams.num_warmup_steps
    num_cosine_steps = num_training_steps - num_warmup_steps

    if num_cosine_steps <= 0:
        raise ValueError("Number of warmup steps is greater than or equal to total training steps.")

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=num_warmup_steps)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_cosine_steps, eta_min=hparams.end_syncnet_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler],milestones=[num_warmup_steps])

    if args.pretrained_model_path is not None:
        # Load from a pretrained model in .pt
        model = load_pretrained_checkpoint(args.pretrained_model_path, model, rank, world_size)
        if args.partial_finetune:
            partial_finetune(model)
        if rank == 0:
            print("Pretrained checkpoint loaded")
    elif args.checkpoint_path is not None:
         # Load from an intermittent ckpt in .pth
        model, optimizer, scheduler, global_step, global_epoch = load_checkpoint(args.checkpoint_path, model, optimizer, scheduler, rank)
        if args.partial_finetune:
            partial_finetune(model)
        if rank == 0:
            print("Checkpoint loaded")
    else:
        if rank == 0:
            print("Training from scratch")

    while global_epoch < nepochs:
        train_sampler.set_epoch(global_epoch)
        test_sampler.set_epoch(global_epoch)

        running_loss = 0.
        running_loss_row = 0.
        running_loss_col = 0.

        running_p1_row = 0.0
        running_p1_col = 0.0
        running_p5_row = 0.0
        running_p5_col = 0.0

        if rank == 0:
            prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        else:
            prog_bar = enumerate(train_data_loader)

        for step, batch_sample in prog_bar:

            if global_step % hparams.syncnet_eval_interval == 0 or global_step == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, device, model, args, hparams)
                    
            model.train()
            optimizer.zero_grad()

            video = batch_sample['video'].to(device)
            audio = batch_sample['audio'].to(device)

            mel, _, _, _ = wav2filterbanks(audio)
            mel_chunk = torch.split(mel, split_size_or_sections=hparams.mel_chunk_size, dim=1)
            mel_chunk = torch.stack(mel_chunk, dim=1)
            mel_chunk = mel_chunk.permute(0, 1, 3, 2)[:, None]

            vid_chunk = torch.split(video, split_size_or_sections=hparams.vid_chunk_size, dim=2)
            vid_chunk = torch.stack(vid_chunk, dim=2)

            B = vid_chunk.size(0)
            face_sequences = torch.cat([vid_chunk[:, :, i] for i in range(vid_chunk.size(2))], dim=0)
            audio_sequences = torch.cat([mel_chunk[:, :, i] for i in range(mel_chunk.size(2))], dim=0)

            with torch.amp.autocast():
                video_embedding, _ = model.module.forward_vid(face_sequences, return_feats=True)
                audio_embedding = model.module.forward_aud(audio_sequences)

                audio_embedding = torch.split(audio_embedding, B, dim=0)
                audio_embedding = torch.stack(audio_embedding, dim=2)
                audio_embedding = audio_embedding.squeeze(3)

                video_embedding = torch.split(video_embedding, B, dim=0)
                video_embedding = torch.stack(video_embedding, dim=2)
                video_embedding = video_embedding.squeeze(3)

            loss_row, p1_row, p5_row, loss_col, p1_col, p5_col, bce_loss = cosine_loss_attn(audio_embedding, video_embedding, model.module, args, device)
            loss = loss_row + loss_col + hparams.mean_suppression_loss_weight * bce_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            running_loss = loss.item()
            running_loss_row = loss_row.item()
            running_loss_col = loss_col.item()
            running_loss_bce = bce_loss.item()
            running_p1_row = p1_row
            running_p1_col = p1_col
            running_p5_row = p5_row
            running_p5_col = p5_col

            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "train/total_loss": running_loss,
                    "train/loss_row": running_loss_row,
                    "train/loss_col": running_loss_col,
                    "train/loss_bce": running_loss_bce,
                    "train/p1_row": running_p1_row,
                    "train/p1_col": running_p1_col,
                    "train/p1_total": (running_p1_row + running_p1_col) / 2,
                    "train/p5_row": running_p5_row,
                    "train/p5_col": running_p5_col,
                    "train/p5_total": (running_p5_row + running_p5_col) / 2,
                    "train/epoch": global_epoch + 1,
                    "train/step": step + 1,
                    "train/lr": current_lr,
                })

            if rank == 0 and global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, global_epoch, checkpoint_dir)

                prog_bar.set_description(
                    'Epoch [{}/{}] Step [{}/{}] Total loss: {:.3f} | Loss-r: {:.3f}, Loss-c: {:.3f}, Loss-bce: {:.3f}'.format(
                        global_epoch + 1, nepochs, step + 1, len(train_data_loader),
                        running_loss,
                        running_loss_row,
                        running_loss_col,
                        running_loss_bce))

        global_epoch += 1

    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, global_step, global_epoch, checkpoint_dir)
        wandb.finish()
    dist.destroy_process_group()


def eval_model(test_data_loader, device, model, args, hparams):
    model.eval()
    eval_steps = len(test_data_loader)
    print(f'Rank {dist.get_rank()}: Evaluating for {eval_steps} steps')

    losses = []
    p1_row_list = []
    p1_col_list = []
    p5_row_list = []
    p5_col_list = []

    if dist.get_rank() == 0:
        eval_loader = tqdm(test_data_loader, total=len(test_data_loader))
    else:
        eval_loader = test_data_loader

    with torch.no_grad():
        for step, batch_sample in enumerate(eval_loader):
            video = batch_sample['video'].to(device)
            audio = batch_sample['audio'].to(device)

            mel, _, _, _ = wav2filterbanks(audio)
            mel_chunk = torch.split(mel, split_size_or_sections=hparams.mel_chunk_size, dim=1)
            mel_chunk = torch.stack(mel_chunk, dim=1)
            mel_chunk = mel_chunk.permute(0, 1, 3, 2)[:, None]

            vid_chunk = torch.split(video, split_size_or_sections=hparams.vid_chunk_size, dim=2)
            vid_chunk = torch.stack(vid_chunk, dim=2)

            B = vid_chunk.size(0)
            face_sequences = torch.cat([vid_chunk[:, :, i] for i in range(vid_chunk.size(2))], dim=0)
            audio_sequences = torch.cat([mel_chunk[:, :, i] for i in range(mel_chunk.size(2))], dim=0)

            with torch.amp.autocast():
                video_embedding, _ = model.module.forward_vid(face_sequences, return_feats=True)
                audio_embedding = model.module.forward_aud(audio_sequences)

                audio_embedding = torch.split(audio_embedding, B, dim=0)
                audio_embedding = torch.stack(audio_embedding, dim=2)
                audio_embedding = audio_embedding.squeeze(3)

                video_embedding = torch.split(video_embedding, B, dim=0)
                video_embedding = torch.stack(video_embedding, dim=2)
                video_embedding = video_embedding.squeeze(3)

            loss_row, p1_row, p5_row, loss_col, p1_col, p5_col, bce_loss = cosine_loss_attn(
                audio_embedding, video_embedding, model.module, args, device)

            loss = loss_row + loss_col + hparams.mean_suppression_loss_weight * bce_loss
            losses.append(loss.item())
            p1_row_list.append(p1_row)
            p1_col_list.append(p1_col)
            p5_row_list.append(p5_row)
            p5_col_list.append(p5_col)

    avg_loss = sum(losses) / len(losses)
    avg_p1_row = sum(p1_row_list) / len(p1_row_list)
    avg_p1_col = sum(p1_col_list) / len(p1_col_list)
    avg_p5_row = sum(p5_row_list) / len(p5_row_list)
    avg_p5_col = sum(p5_col_list) / len(p5_col_list)

    avg_loss_tensor = torch.tensor(avg_loss, device=device)
    avg_p1_row_tensor = torch.tensor(avg_p1_row, device=device)
    avg_p1_col_tensor = torch.tensor(avg_p1_col, device=device)
    avg_p5_row_tensor = torch.tensor(avg_p5_row, device=device)
    avg_p5_col_tensor = torch.tensor(avg_p5_col, device=device)

    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_p1_row_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_p1_col_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_p5_row_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_p5_col_tensor, op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size()

    avg_loss = avg_loss_tensor.item() / world_size
    avg_p1_row = avg_p1_row_tensor.item() / world_size
    avg_p1_col = avg_p1_col_tensor.item() / world_size
    avg_p5_row = avg_p5_row_tensor.item() / world_size
    avg_p5_col = avg_p5_col_tensor.item() / world_size

    avg_p1_total = (avg_p1_row + avg_p1_col) / 2
    avg_p5_total = (avg_p5_row + avg_p5_col) / 2

    if dist.get_rank() == 0:
        wandb.log({
            "eval/total_loss": avg_loss,
            "eval/p1_row": avg_p1_row,
            "eval/p1_col": avg_p1_col,
            "eval/p1_total": avg_p1_total,
            "eval/p5_row": avg_p5_row,
            "eval/p5_col": avg_p5_col,
            "eval/p5_total": avg_p5_total,
        })


def save_checkpoint(model, optimizer, scheduler, global_step, global_epoch, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step{global_step:09d}.pth")
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        'global_epoch': global_epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def load_checkpoint(path, model, optimizer, scheduler, rank):
    print(f"Loading checkpoint from: {path}")
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(path, map_location=map_location)

    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(rank)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scheduler.last_epoch = checkpoint['global_step']

    global_step = checkpoint['global_step']
    global_epoch = checkpoint['global_epoch']

    return model, optimizer, scheduler, global_step, global_epoch


def load_pretrained_checkpoint(path, model, rank, world_size):
    print(f"Load checkpoint from: {path}")
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    s = _load(path, map_location)
    new_s = {}
    for k, v in s.items():
        if world_size > 1:
            new_s['module.' + k] = v if not k.startswith('module.') else v
        else:
            new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s, strict=False)
    return model


def partial_finetune(model):
    """
    Freeze all model parameters except for specific submodules to enable partial fine-tuning.
    The following components remain trainable:
      - ff_lip: feed-forward network for lip features
      - ff_aud: feed-forward network for audio features
      - vid_adapter: adapter module for visual features
      - aud_adapter: adapter module for audio features
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.module.ff_lip.parameters():
        param.requires_grad = True
    for param in model.module.ff_aud.parameters():
        param.requires_grad = True
    for param in model.module.vid_adapter.parameters():
        param.requires_grad = True
    for param in model.module.aud_adapter.parameters():
        param.requires_grad = True


def _load(checkpoint_path, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default=None, required=False, type=str)
    parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
    parser.add_argument('--pretrained_model_path', help='Path to pretrained model weights', default=None, type=str)
    parser.add_argument('--height', type=int, default=270)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_gpu', type=int, default=torch.cuda.device_count(), required=False, help='Number of GPUs')
    parser.add_argument('--similarity_type', default="max", required=False, help="Spatial similarity to be used [mean or max]")
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--partial_finetune', action='store_true', help='Partial Finetuning or Full Finetuning')
    parser.add_argument('--scale_mean', action='store_true', help='Scaling mean')
    parser.add_argument('--synthetic', action='store_true', help='Whether use synthetic data for training')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    mp.spawn(main, args=(args.n_gpu, args), nprocs=args.n_gpu, join=True)
    