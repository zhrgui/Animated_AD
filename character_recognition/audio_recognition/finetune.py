"""
Fine-tune the LWTNet audio-visual synchronisation model.

The starting point is a pretrained checkpoint - the released AVObjects LWTNet,
trained on real talking faces - which is adapted to animated video. By default
only the projection heads and the temporal adapters are trained and the encoders
are left frozen, which is what makes this work on the amount of animated data we
have; set `model.trainable` to `[]` for a full fine-tune.

    # adapter-only fine-tune on animated shots, 4 GPUs
    python audio_recognition/finetune.py --config audio_recognition/config/finetune_animated.yaml

    # override anything from the command line
    python audio_recognition/finetune.py --config <cfg> \
        data.data_folder=/datasets/simpsons/vid train.lr=5e-5 wandb.enabled=false

Multi-GPU is handled internally (one process per GPU via torch.multiprocessing),
so the script is launched as a plain python program, not through torchrun.
"""

import argparse
import os

# librosa pulls in numba, which wants a writable cache directory.
os.environ.setdefault('NUMBA_CACHE_DIR', '')

import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils import data as data_utils
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.audio import audio_opts, wav2filterbanks
from data.datasets import DATASETS, build_dataset
from model.checkpoint import (count_parameters, freeze_except, load_model_params,
                              load_training_state, save_training_state, set_train_mode,
                              unwrap)
from model.losses import sync_contrastive_loss
from model.lwtnet import LWTNet_temporal_adapter

warnings.filterwarnings("ignore", category=DeprecationWarning)


DEFAULT_CONFIG = {
    'seed': 0,

    'model': {
        # Weights to start from: the pretrained LWTNet .pt, or one of our own
        # .pth checkpoints. Anything the checkpoint does not contain (the
        # adapters, when starting from a plain LWTNet) keeps its initialisation.
        'pretrained': None,
        # Submodules to train; [] or null trains the whole network.
        'trainable': ['ff_lip', 'ff_aud', 'vid_adapter', 'aud_adapter'],
        # Keep frozen BatchNorm layers in eval mode so a partial fine-tune does
        # not drift the running statistics of the frozen encoders.
        'frozen_bn_eval': True,
    },

    'data': {
        'dataset': 'lrs3',   # lrs2 | lrs3 | synthetic | cmd_animated | simpsons
        'data_folder': None,         # cmd_animated / simpsons
        'movie_title': None,         # cmd_animated, null for every movie
        'val_ratio': 0.1,            # lrs2 / cmd_animated / simpsons
        'lrs2_path': None,           # lrs2
        'lrs3_path': None,           # lrs3 / synthetic
        'synthetic_path': None,      # synthetic
        'synthetic_split': None,     # synthetic
        'mixing_ratio': 0.5,         # synthetic
        'height': 270,
        'width': 480,
        'fps': 25,
        'sample_rate': 16000,
        'num_frames': 150,           # 6s window
        'vid_chunk_size': 15,        # 0.6s
        'mel_chunk_size': 60,        # 0.6s
        'num_workers': 8,
    },

    'train': {
        'epochs': 5,
        'batch_size': 16,            # total across all GPUs
        'lr': 1.0e-4,
        'min_lr': 1.0e-6,
        'warmup_steps': 1000,
        'similarity': 'max',         # max | mean
        'scale_mean': True,
        'mean_suppression_weight': 5.0,
        'grad_clip': None,
        'amp': True,
    },

    'eval': {
        'interval': 5000,            # in optimisation steps
    },

    'checkpoint': {
        'dir': None,
        'interval': 5000,
        'resume': None,              # training state to continue from
    },

    'wandb': {
        'enabled': True,
        'project': 'avobjects_finetune',
        'entity': None,          # null = the account's default entity
        'name': None,
    },

    'dist': {
        'master_addr': 'localhost',
        'master_port': '12355',
    },
}


# --------------------------------------------------------------------- config

def load_config(path, overrides):
    base = OmegaConf.create(DEFAULT_CONFIG)
    # Struct mode: a key that is not in DEFAULT_CONFIG is a typo, and a silently
    # ignored config key is a long afternoon.
    OmegaConf.set_struct(base, True)

    cfg = OmegaConf.merge(base, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    validate_config(cfg)
    return cfg


REQUIRED_DATA_KEYS = {
    'lrs2': ['lrs2_path'],
    'lrs3': ['lrs3_path'],
    'synthetic': ['lrs3_path', 'synthetic_path', 'synthetic_split'],
    'cmd_animated': ['data_folder'],
    'simpsons': ['data_folder'],
}


def validate_config(cfg):
    if cfg.checkpoint.dir is None:
        raise ValueError("checkpoint.dir is not set: nowhere to write the fine-tuned weights")

    if cfg.data.dataset not in DATASETS:
        raise ValueError(f"Unknown data.dataset {cfg.data.dataset!r}, "
                         f"expected one of {sorted(DATASETS)}")

    unset = [k for k in REQUIRED_DATA_KEYS[cfg.data.dataset] if cfg.data.get(k) is None]
    if unset:
        raise ValueError(f"data.dataset={cfg.data.dataset} needs {', '.join('data.' + k for k in unset)}")

    if cfg.data.num_frames % cfg.data.vid_chunk_size:
        raise ValueError(
            f"data.num_frames ({cfg.data.num_frames}) must be a multiple of "
            f"data.vid_chunk_size ({cfg.data.vid_chunk_size})")

    # The two streams are matched chunk by chunk, so they have to produce the
    # same number of chunks from a window.
    mel_frames = cfg.data.num_frames * cfg.data.sample_rate // cfg.data.fps // audio_opts['hop_length']
    n_vid = cfg.data.num_frames // cfg.data.vid_chunk_size
    n_mel = mel_frames // cfg.data.mel_chunk_size

    if mel_frames % cfg.data.mel_chunk_size:
        raise ValueError(
            f"a {cfg.data.num_frames}-frame window is {mel_frames} mel frames, which is not a "
            f"multiple of data.mel_chunk_size ({cfg.data.mel_chunk_size})")
    if n_vid != n_mel:
        raise ValueError(
            f"video and audio chunk counts differ ({n_vid} vs {n_mel}): adjust "
            f"data.vid_chunk_size / data.mel_chunk_size")


# ----------------------------------------------------------------- distributed

def setup(rank, world_size, cfg):
    os.environ['MASTER_ADDR'] = str(cfg.dist.master_addr)
    os.environ['MASTER_PORT'] = str(cfg.dist.master_port)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def all_reduce_mean(metrics, device):
    """Average a dict of python floats over the ranks."""
    keys = sorted(metrics)
    values = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {k: v.item() for k, v in zip(keys, values)}


# ------------------------------------------------------------------- one step

def chunk_batch(video, audio, cfg):
    """
    Cut a window into the fixed-size chunks the two encoders expect.

    :param video: B x C x T x H x W
    :param audio: B x (T * sample_rate / fps)
    :return: (B*n) x C x t x H x W video chunks, (B*n) x 1 x F x m mel chunks, B
    """
    mel, _, _, _ = wav2filterbanks(audio)
    mel_chunk = torch.stack(torch.split(mel, cfg.data.mel_chunk_size, dim=1), dim=1)
    mel_chunk = mel_chunk.permute(0, 1, 3, 2)[:, None]

    vid_chunk = torch.stack(torch.split(video, cfg.data.vid_chunk_size, dim=2), dim=2)

    batch_size = vid_chunk.size(0)
    face_sequences = torch.cat([vid_chunk[:, :, i] for i in range(vid_chunk.size(2))], dim=0)
    audio_sequences = torch.cat([mel_chunk[:, :, i] for i in range(mel_chunk.size(2))], dim=0)
    return face_sequences, audio_sequences, batch_size


def forward_batch(model, batch_sample, cfg, device, autocast_enabled):
    """Encode a batch and score every (video chunk, audio chunk) pair against it."""
    video = batch_sample['video'].to(device, non_blocking=True)
    audio = batch_sample['audio'].to(device, non_blocking=True)

    face_sequences, audio_sequences, batch_size = chunk_batch(video, audio, cfg)

    # Note: the DDP wrapper is called, not model.module - otherwise the gradients
    # are never synchronised across ranks.
    with torch.amp.autocast('cuda', enabled=autocast_enabled):
        vid_emb, aud_emb = model(face_sequences, audio_sequences)

    aud_emb = torch.stack(torch.split(aud_emb, batch_size, dim=0), dim=2).squeeze(3)
    vid_emb = torch.stack(torch.split(vid_emb, batch_size, dim=0), dim=2).squeeze(3)

    # The loss runs in fp32: binary_cross_entropy is not autocast-safe and the
    # similarity matrix is small enough that this costs nothing.
    metrics = sync_contrastive_loss(vid_emb.float(), aud_emb.float(),
                                    unwrap(model).logits_scale,
                                    similarity=cfg.train.similarity,
                                    scale_mean=cfg.train.scale_mean)

    metrics['loss'] = (metrics['loss_row'] + metrics['loss_col']
                       + cfg.train.mean_suppression_weight * metrics['loss_bce'])
    return metrics


def scalar_metrics(metrics):
    """Detach the loss terms so they can be logged / reduced."""
    out = {k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()}
    out['p1_total'] = (out['p1_row'] + out['p1_col']) / 2
    out['p5_total'] = (out['p5_row'] + out['p5_col']) / 2
    return out


# ----------------------------------------------------------------------- train

def main(rank, world_size, cfg):
    setup(rank, world_size, cfg)
    is_master = rank == 0

    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    torch.backends.cudnn.benchmark = True

    device = torch.device(f"cuda:{rank}")

    if cfg.train.batch_size % world_size:
        raise ValueError(f"train.batch_size ({cfg.train.batch_size}) is not divisible "
                         f"by the number of GPUs ({world_size})")
    batch_size = cfg.train.batch_size // world_size

    use_wandb = bool(cfg.wandb.enabled) and is_master
    if use_wandb:
        import wandb
        wandb.login()
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                   name=cfg.wandb.name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    # ---------------------------------------------------------------- model
    model = LWTNet_temporal_adapter().to(device)

    # Load the pretrained weights before wrapping: whatever the checkpoint does
    # not cover (the adapters) stays at its initialisation.
    if cfg.model.pretrained is not None:
        missing = load_model_params(model, cfg.model.pretrained, verbose=is_master)
        if is_master:
            print(f"Initialised from {cfg.model.pretrained} "
                  f"({len(missing)} tensors trained from scratch)")
    elif cfg.checkpoint.resume is None and is_master:
        print("No pretrained weights given, training from scratch")

    freeze_except(model, cfg.model.trainable)
    if is_master:
        trainable, total = count_parameters(model)
        print(f"Training {trainable:,} of {total:,} parameters "
              f"({cfg.model.trainable or 'all modules'})")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # ----------------------------------------------------------------- data
    train_dataset = build_dataset(cfg.data, 'train')
    val_dataset = build_dataset(cfg.data, 'val')
    if is_master:
        print(f"{cfg.data.dataset}: {len(train_dataset)} train / {len(val_dataset)} val clips")
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError(f"Empty {cfg.data.dataset} split, check data.* in the config")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    loader_kwargs = dict(batch_size=batch_size, num_workers=cfg.data.num_workers,
                         pin_memory=True, persistent_workers=cfg.data.num_workers > 0)
    train_data_loader = data_utils.DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
    val_data_loader = data_utils.DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

    # ------------------------------------------------------- optimiser / lr
    # Built after freezing, so that a partial fine-tune really does leave the
    # frozen weights alone.
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg.train.lr)

    num_training_steps = cfg.train.epochs * len(train_data_loader)
    num_warmup_steps = min(cfg.train.warmup_steps, max(num_training_steps - 1, 0))
    num_cosine_steps = num_training_steps - num_warmup_steps

    if num_cosine_steps <= 0:
        raise ValueError(
            f"train.warmup_steps ({cfg.train.warmup_steps}) leaves no steps for the cosine "
            f"schedule ({num_training_steps} total)")

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=num_warmup_steps)
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_cosine_steps, eta_min=cfg.train.min_lr)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.train.amp)

    global_step, global_epoch = 0, 0
    if cfg.checkpoint.resume is not None:
        global_step, global_epoch = load_training_state(
            cfg.checkpoint.resume, model, optimizer, scheduler,
            map_location={'cuda:0': f'cuda:{rank}'}, device=rank)
        if is_master:
            print(f"Resumed {cfg.checkpoint.resume} at epoch {global_epoch}, step {global_step}")

    if is_master:
        os.makedirs(cfg.checkpoint.dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.checkpoint.dir, 'config.yaml'))

    best_val_loss = float('inf')

    # ------------------------------------------------------------ the loop
    while global_epoch < cfg.train.epochs:
        train_sampler.set_epoch(global_epoch)
        val_sampler.set_epoch(global_epoch)

        prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                        disable=not is_master)

        for step, batch_sample in prog_bar:

            if global_step % cfg.eval.interval == 0:
                val_metrics = evaluate(val_data_loader, model, cfg, device, is_master)
                if is_master:
                    print(f"[eval @ step {global_step}] loss {val_metrics['loss']:.3f} "
                          f"p1 {val_metrics['p1_total']:.1f} p5 {val_metrics['p5_total']:.1f}")
                    if use_wandb:
                        wandb.log({f"eval/{k}": v for k, v in val_metrics.items()},
                                  step=global_step)
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        save_training_state(
                            os.path.join(cfg.checkpoint.dir, 'best.pth'), model, optimizer,
                            scheduler, global_step, global_epoch,
                            OmegaConf.to_container(cfg, resolve=True))

            set_train_mode(model, frozen_bn_eval=cfg.model.frozen_bn_eval)
            optimizer.zero_grad(set_to_none=True)

            metrics = forward_batch(model, batch_sample, cfg, device, cfg.train.amp)

            scaler.scale(metrics['loss']).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if is_master:
                logs = scalar_metrics(metrics)
                prog_bar.set_description(
                    'Epoch [{}/{}] Step [{}/{}] loss {:.3f} | row {:.3f}, col {:.3f}, bce {:.3f}'.format(
                        global_epoch + 1, cfg.train.epochs, step + 1, len(train_data_loader),
                        logs['loss'], logs['loss_row'], logs['loss_col'], logs['loss_bce']))

                if use_wandb:
                    wandb.log({**{f"train/{k}": v for k, v in logs.items()},
                               "train/epoch": global_epoch + 1,
                               "train/lr": optimizer.param_groups[0]['lr']},
                              step=global_step)

                if global_step % cfg.checkpoint.interval == 0:
                    path = save_training_state(
                        os.path.join(cfg.checkpoint.dir, f"checkpoint_step{global_step:09d}.pth"),
                        model, optimizer, scheduler, global_step, global_epoch,
                        OmegaConf.to_container(cfg, resolve=True))
                    print("Saved checkpoint:", path)

        global_epoch += 1

    val_metrics = evaluate(val_data_loader, model, cfg, device, is_master)
    if is_master:
        print(f"[final eval] loss {val_metrics['loss']:.3f} p1 {val_metrics['p1_total']:.1f}")
        path = save_training_state(os.path.join(cfg.checkpoint.dir, 'last.pth'), model, optimizer,
                                   scheduler, global_step, global_epoch,
                                   OmegaConf.to_container(cfg, resolve=True))
        print("Saved checkpoint:", path)
        if use_wandb:
            wandb.finish()

    dist.destroy_process_group()


@torch.no_grad()
def evaluate(val_data_loader, model, cfg, device, is_master):
    model.eval()

    totals = None
    n_batches = 0

    for batch_sample in tqdm(val_data_loader, desc='eval', leave=False, disable=not is_master):
        logs = scalar_metrics(forward_batch(model, batch_sample, cfg, device, cfg.train.amp))
        totals = logs if totals is None else {k: totals[k] + v for k, v in logs.items()}
        n_batches += 1

    if not n_batches:
        raise ValueError("The validation split produced no batches")

    return all_reduce_mean({k: v / n_batches for k, v in totals.items()}, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune LWTNet on animated (or real) audio-visual data.")
    parser.add_argument('--config', required=True, type=str,
                        help='YAML config, see audio_recognition/config/')
    parser.add_argument('--n_gpu', type=int, default=torch.cuda.device_count(),
                        help='Number of GPUs to train on')
    parser.add_argument('overrides', nargs='*',
                        help='Config overrides, e.g. train.lr=5e-5 wandb.enabled=false')
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    if args.n_gpu < 1:
        raise SystemExit("No GPU available: this fine-tune needs at least one.")

    os.makedirs(cfg.checkpoint.dir, exist_ok=True)
    mp.spawn(main, args=(args.n_gpu, cfg), nprocs=args.n_gpu, join=True)
