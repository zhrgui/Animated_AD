"""
Datasets for the audio-visual synchronisation model.

Training samples are a random fixed-length window of a clip together with the
matching audio; the variants differ only in *which* clips they draw from and in
how a clip path maps to its wav file, so that logic lives in `AVWindowDataset`
and each dataset only supplies those two things.

`DemoDataset` is the odd one out: it is the inference path and returns a whole
clip rather than a window.
"""

import os
import pickle
import random
import subprocess

import numpy as np
import torch
from torch.utils import data

from glob import glob
from decord import VideoReader, cpu

from data.audio import load_wav

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --------------------------------------------------------------- file listings

def _get_image_list(dataset, split, path, cache_dir='.'):
    """
    List the clips under `path`, split 95/5 into train/val, and cache the result
    so that every worker and every run sees the same split.
    """
    pkl_file = os.path.join(cache_dir, 'filenames_{}_{}.pkl'.format(dataset, split))
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as p:
            return pickle.load(p)

    filelist = glob(path)
    filelist = [f for f in filelist if f.endswith(('.mp4', '.avi'))]
    random.shuffle(filelist)

    if split == 'train':
        filelist = filelist[:int(.95 * len(filelist))]
    else:
        filelist = filelist[int(.95 * len(filelist)):]

    with open(pkl_file, 'wb') as p:
        pickle.dump(filelist, p, protocol=pickle.HIGHEST_PROTOCOL)

    return filelist


def _holdout_split(files, split, val_ratio=0.1):
    """
    Deterministic train/val split of a glob listing.

    The list is sorted first so that the two splits do not depend on the order
    the filesystem happened to return, i.e. so that a validation clip stays a
    validation clip across runs and across ranks.
    """
    files = sorted(files)
    n_val = max(1, int(len(files) * val_ratio)) if files else 0
    return files[:n_val] if split == 'val' else files[n_val:]


def lrs3_wav_path(video_path):
    """LRS3 layout: .../mp4/.../xyz.mp4 -> .../wav/.../xyz.wav"""
    return video_path.replace("mp4", "wav")


def lrs2_wav_path(video_path):
    """LRS2 layout: .../vid/{id}/xyz.mp4 -> .../wav/clean/{id}/xyz.wav"""
    return video_path.replace("/vid/", "/wav/clean/").replace(".mp4", ".wav")


def sibling_wav_path(video_path):
    """CMDAM layout: .../vid/.../xyz.mp4 -> .../aud/.../xyz.wav"""
    return video_path.replace("/vid/", "/aud/").replace(".mp4", ".wav")


# ------------------------------------------------------------- training window

class AVWindowDataset(data.Dataset):
    """
    A random `num_frames`-long window of a random clip, with its audio.

    Frames are handed over in 0..255, the range the model is fed at inference
    too (`model.sync_scorer.SyncScorerBase.encode`).
    """

    wav_path = staticmethod(sibling_wav_path)

    def __init__(self, files, num_frames=150, height=270, width=480, fps=25,
                 sample_rate=16000):
        self.files = files
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # The index is only used to size the epoch: a sample is drawn at random
        # and re-drawn until one yields a long enough, readable window.
        while True:
            index = random.randint(0, len(self.files) - 1)
            fname = self.files[index]
            if "mp4" not in fname:
                fname = fname + ".mp4"

            video, start_frame = self.load_frames(fname)
            if video is None:
                continue

            audio, aud_fact = self.load_audio(fname, start_frame)
            if audio is None:
                continue

            if not aud_fact * video.shape[1] == audio.shape[0]:
                continue

            video = torch.FloatTensor(np.array(video))
            audio = torch.FloatTensor(audio)

            return {'video': video, 'audio': audio, 'sample': fname}

    def load_frames(self, fname):
        try:
            vr = VideoReader(fname, ctx=cpu(0), width=self.width, height=self.height)
        except Exception:
            return None, None

        try:
            if len(vr) - self.num_frames - 1 <= 0:
                return None, None

            start_frame = random.choice(range(0, len(vr) - self.num_frames - 1))
            end_frame = start_frame + self.num_frames

            selected_frames = range(start_frame, end_frame)
            input_frames = vr.get_batch(selected_frames).asnumpy()
        except Exception:
            return None, None

        input_frames = np.asarray(input_frames)
        input_frames = np.transpose(input_frames, (3, 0, 1, 2))  # t h w c -> c t h w

        return input_frames, start_frame

    def load_audio(self, fname, start_frame):
        try:
            audio = load_wav(self.wav_path(fname)).astype('float32')
        except Exception:
            return None, None

        aud_fact = int(np.round(self.sample_rate / self.fps))
        audio_window = audio[aud_fact * start_frame: aud_fact * (start_frame + self.num_frames)]
        audio_window = np.array(audio_window)

        return audio_window, aud_fact


class DataGenerator_Fulldata(AVWindowDataset):
    """LRS3 pretraining clips."""

    wav_path = staticmethod(lrs3_wav_path)

    def __init__(self, lrs3_pretrain_path, split='train', **kwargs):
        files = _get_image_list('lrs3_pretrain', split, lrs3_pretrain_path)
        super().__init__(files, **kwargs)


class DataGenerator_LRS2(AVWindowDataset):
    """
    LRS2 pretraining clips.

    Two things differ from LRS3. The audio does not sit next to the video but
    under `wav/clean/` mirroring `vid/`, so `lrs3_wav_path` - which on this
    layout only rewrites the extension - would look for a wav that is not there.
    And the split is the deterministic `_holdout_split` rather than the cached
    shuffle: sorting groups the clips by speaker directory, so the held-out
    clips come from speakers that are not trained on, and every rank derives the
    same split without a pickle to race over.
    """

    wav_path = staticmethod(lrs2_wav_path)

    def __init__(self, lrs2_path, split='train', val_ratio=0.1, **kwargs):
        files = [f for f in glob(lrs2_path) if f.endswith(('.mp4', '.avi'))]
        files = _holdout_split(files, split, val_ratio)
        random.shuffle(files)
        super().__init__(files, **kwargs)


class DataGenerator_Synthetic(AVWindowDataset):
    """
    LRS3 clips mixed with the synthetic two-speaker clips built by
    `preprocess/generate_synthetic.py`, in a fixed ratio.
    """

    wav_path = staticmethod(lrs3_wav_path)

    def __init__(self, lrs3_pretrain_path, synthetic_path, synthetic_split,
                 mixing_ratio, split='train', **kwargs):
        lrs3_files = _get_image_list('lrs3_pretrain', split, lrs3_pretrain_path)
        synthetic_files = _get_image_list(synthetic_split, split, synthetic_path)

        random.shuffle(lrs3_files)
        random.shuffle(synthetic_files)

        total_size = len(lrs3_files)
        num_synthetic = int(total_size * mixing_ratio)
        num_lrs3 = total_size - num_synthetic

        files = lrs3_files[:num_lrs3] + synthetic_files[:num_synthetic]
        random.shuffle(files)

        super().__init__(files, **kwargs)


class CMD_Animated_Dataset(AVWindowDataset):
    """Per-shot animated clips, optionally restricted to a single movie."""

    def __init__(self, data_folder, movie_title=None, split='train',
                 val_ratio=0.1, **kwargs):
        if movie_title is not None:
            files = glob(os.path.join(data_folder, movie_title) + "/*/*")
        else:
            files = glob(data_folder + "/*/*/*")

        files = _holdout_split(files, split, val_ratio)
        random.shuffle(files)
        super().__init__(files, **kwargs)


class Simpsons_Dataset(AVWindowDataset):
    """Per-shot Simpsons clips, laid out as <data_folder>/<season>/<episode>/<shot>.mp4"""

    def __init__(self, data_folder, split='train', val_ratio=0.1, **kwargs):
        files = _holdout_split(glob(data_folder + "/*/*/*"), split, val_ratio)
        random.shuffle(files)
        super().__init__(files, **kwargs)


class CMD_Animated_DemoDataset(AVWindowDataset):
    """A single animated clip, for a quick look at what the model does with it."""

    def __init__(self, video_path, **kwargs):
        super().__init__([video_path], **kwargs)


DATASETS = {
    'lrs2': DataGenerator_LRS2,
    'lrs3': DataGenerator_Fulldata,
    'synthetic': DataGenerator_Synthetic,
    'cmd_animated': CMD_Animated_Dataset,
    'simpsons': Simpsons_Dataset,
}


def build_dataset(cfg, split):
    """
    Build the dataset named by `cfg.dataset` for `split` ('train' or 'val').

    `cfg` is the `data` section of a fine-tuning config: the window/decoding
    options are shared by every dataset, the rest are the arguments of the one
    that was selected.
    """
    if cfg.dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}, expected one of {sorted(DATASETS)}")

    common = dict(
        num_frames=cfg.num_frames,
        height=cfg.height,
        width=cfg.width,
        fps=cfg.fps,
        sample_rate=cfg.sample_rate,
    )

    if cfg.dataset == 'lrs2':
        return DataGenerator_LRS2(cfg.lrs2_path, split=split,
                                  val_ratio=cfg.get('val_ratio', 0.1), **common)
    if cfg.dataset == 'lrs3':
        return DataGenerator_Fulldata(cfg.lrs3_path, split=split, **common)
    if cfg.dataset == 'synthetic':
        return DataGenerator_Synthetic(cfg.lrs3_path, cfg.synthetic_path,
                                       cfg.synthetic_split, cfg.mixing_ratio,
                                       split=split, **common)
    if cfg.dataset == 'cmd_animated':
        return CMD_Animated_Dataset(cfg.data_folder,
                                    movie_title=cfg.get('movie_title', None),
                                    split=split,
                                    val_ratio=cfg.get('val_ratio', 0.1),
                                    **common)
    return Simpsons_Dataset(cfg.data_folder, split=split,
                            val_ratio=cfg.get('val_ratio', 0.1), **common)


# ------------------------------------------------------------------- inference

class DemoDataset(data.Dataset):
    """A whole clip, resampled to 25 fps with its audio extracted to 16 kHz mono."""

    def __init__(self, video_path, resize, fps, sample_rate):
        self.resize = resize
        self.fps = fps
        self.sample_rate = sample_rate
        self.all_vids = [video_path]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.all_vids)

    def __getitem__(self, index):
        'Generates one sample of data'
        assert index < self.__len__()

        vid_path = self.all_vids[index]
        vid_name, _ = os.path.splitext(vid_path)

        # -- reencode video to 25 fps
        vid_path_orig = vid_path
        vid_path_25fps = vid_name + '_25fps.mp4'

        cmd = [
            "ffmpeg",
            "-threads", "1",
            "-loglevel", "error",
            "-y",
            "-i", vid_path_orig,
            "-an",
            "-r", "25",
            vid_path_25fps
        ]
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        video, size_info = self.__load_video__(vid_path_25fps)

        # -- extract the audio as 16 kHz mono
        aud_path = vid_name + '.wav'
        cmd = [
            "ffmpeg",
            "-threads", "1",
            "-loglevel", "error",
            "-y",
            "-i", vid_path_orig,
            "-async", "1",
            "-ac", "1",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            aud_path
        ]
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        audio = load_wav(aud_path).astype('float32')

        fps = self.fps
        aud_fact = int(np.round(self.sample_rate / fps))
        audio, video = self.trunkate_audio_and_video(video, audio, aud_fact)
        assert aud_fact * video.shape[0] == audio.shape[0]

        audio = np.array(audio)
        video = video.transpose([3, 0, 1, 2])  # t c h w -> c t h w

        return {
            'video': video,
            'audio': audio,
            'size': size_info,
            'sample': vid_path,
        }

    def __load_video__(self, video_file, width=480, height=270):
        try:
            vr = VideoReader(video_file, ctx=cpu(0), width=width, height=height)

            default_vr = VideoReader(video_file, ctx=cpu(0))
            frame0 = default_vr[0]
            original_height, original_width, _ = frame0.shape
        except Exception as exc:
            raise RuntimeError(f"Could not load the input video file {video_file}") from exc

        input_frames = []
        for k in range(len(vr)):
            input_frames.append(vr[k].asnumpy())

        frames = np.asarray(input_frames)
        return frames.astype('float32'), {"original_size": (original_width, original_height),
                                          "new_size": (width, height)}

    def trunkate_audio_and_video(self, video, aud_feats, aud_fact):
        # make audio exactly divisible by video frames
        aud_cutoff = min(video.shape[0], int(aud_feats.shape[0] / aud_fact))

        aud_feats = aud_feats[:aud_cutoff * aud_fact]
        aud_in_frames = aud_feats.shape[0] // aud_fact

        min_len = min(aud_in_frames, video.shape[0])

        # --- trunkate all to min
        video = video[:min_len]
        aud_feats = aud_feats[:min_len * aud_fact]

        if not aud_feats.shape[0] // aud_fact == video.shape[0]:
            raise ValueError(
                "Audio and video could not be aligned: {} audio frames vs {} video frames".format(
                    aud_feats.shape[0] // aud_fact, video.shape[0]))

        return aud_feats, video
