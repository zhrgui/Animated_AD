import os
import pickle
import random
import numpy as np
import torch
from torch.utils import data

from glob import glob
from decord import VideoReader, cpu
from load_audio import load_wav

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def _get_image_list(dataset, split, path):
	pkl_file = 'filenames_{}_{}.pkl'.format(dataset, split)
	if os.path.exists(pkl_file):
		with open(pkl_file, 'rb') as p:
			return pickle.load(p)
	else:
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
    

def _get_all_files(split, lrs3_pretrain_path):
    filelist_lrs3_pretrain = _get_image_list('lrs3_pretrain', split, lrs3_pretrain_path)
    filelist = filelist_lrs3_pretrain
    return filelist


def _get_all_files_(dataset_name, split, lrs3_pretrain_path):
    filelist_lrs3_pretrain = _get_image_list(dataset_name, split, lrs3_pretrain_path)
    filelist = filelist_lrs3_pretrain
    return filelist


class DemoDataset(data.Dataset):
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

        # -- load video
        vid_path_orig = vid_path
        # vid_path_25fps = os.path.join(vid_name, '_25fps.mp4')
        vid_path_25fps = vid_name + '_25fps.mp4'

        # -- reencode video to 25 fps
        import subprocess
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
        # print('Resampling {} to 25 fps'.format(vid_path_orig))
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        video, size_info = self.__load_video__(vid_path_25fps)
        # aud_path = os.path.join(vid_name, '.wav')
        aud_path = vid_name + '.wav'
        # if not os.path.exists(aud_path):
        import subprocess
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

        out_dict = {
            'video': video,
            'audio': audio,
            'size': size_info,
            'sample': vid_path}
        return out_dict

    def __load_video__(self, video_file, width=480, height=270):
        try:
            vr = VideoReader(video_file, ctx=cpu(0), width=width, height=height)

            default_vr = VideoReader(video_file, ctx=cpu(0))
            frame0 = default_vr[0]
            original_height, original_width, _ = frame0.shape
        except:
            print("Oops! Could not load the input video file")
            return None

        input_frames = []
        for k in range(len(vr)):
            input_frames.append(vr[k].asnumpy())

        frames = np.asarray(input_frames)
        return frames.astype('float32'), {"original_size": (original_width, original_height), "new_size": (width, height)}

    def trunkate_audio_and_video(self, video, aud_feats, aud_fact):

        aud_in_frames = aud_feats.shape[0] // aud_fact

        # make audio exactly devisible by video frames
        aud_cutoff = min(video.shape[0], int(aud_feats.shape[0] / aud_fact))

        aud_feats = aud_feats[:aud_cutoff * aud_fact]
        aud_in_frames = aud_feats.shape[0] // aud_fact

        min_len = min(aud_in_frames, video.shape[0])

        # --- trunkate all to min
        video = video[:min_len]
        aud_feats = aud_feats[:min_len * aud_fact]
        if not aud_feats.shape[0] // aud_fact == video.shape[0]:
            import ipdb
            ipdb.set_trace(context=20)

        return aud_feats, video
    

class DataGenerator_Fulldata(data.Dataset):
    def __init__(self, hparams, height=270, width=480, fps=25, sample_rate=16000, test=False):
        self.files = _get_all_files('train', hparams.lrs3_pretrain_path) if not test else _get_all_files('val', hparams.lrs3_pretrain_path)
        self.num_frames = hparams.syncnet_T
        self.height = height
        self.width = width
        self.fps = fps
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        while(1):
            index = random.randint(0, len(self.files) - 1)
            fname = self.files[index]
            # if "avi" not in fname:
            #     fname = fname+".avi"
            if "mp4" not in fname:
                fname = fname+".mp4"

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

            out_dict = {
                'video': video,
                'audio': audio,
                'sample': fname
            }

            return out_dict

    def load_frames(self, fname):
        try:
            vr = VideoReader(fname, ctx=cpu(0), width=self.width, height=self.height)
        except:
            return None, None

        try:
            if len(vr)-self.num_frames-1 <= 0:
                return None, None
            
            start_frame = random.choice(range(0, len(vr)-self.num_frames-1))
            end_frame = start_frame + self.num_frames

            selected_frames = range(start_frame, end_frame)
            input_frames = vr.get_batch(selected_frames).asnumpy()
        except:
            return None, None

        input_frames = np.asarray(input_frames) / 255.

        input_frames = np.transpose(input_frames, (3, 0, 1, 2)) 
        
        return input_frames, start_frame

    def load_audio(self, fname, start_frame):
        try:
            # # lrs3 dataset
            wavpath = fname.replace("mp4", "wav")
            wavpath = wavpath.replace(".mp4", ".wav")
            audio = load_wav(wavpath).astype('float32')

            # lrs2 dataset
            # wavpath = fname.replace("vid", "wav/clean")
            # wavpath = wavpath.replace(".mp4", ".wav")
            # audio = load_wav(wavpath).astype('float32')

        except:
            return None, None

        aud_fact = int(np.round(self.sample_rate / self.fps))
        audio_window = audio[aud_fact*start_frame : aud_fact*(start_frame+self.num_frames)]
        audio_window = np.array(audio_window)

        return audio_window, aud_fact


class DataGenerator_Synthetic(data.Dataset):
    def __init__(self, hparams, height=270, width=480, fps=25, sample_rate=16000, test=False):
        self.lrs3_files = _get_all_files_('lrs3_pretrain', 'train', hparams.lrs3_pretrain_path) if not test else _get_all_files_('lrs3_pretrain', 'val', hparams.lrs3_pretrain_path)
        self.lrs3_synthetic_files = _get_all_files_(hparams.lrs3_synthetic_split, 'train', hparams.lrs3_synthetic_path) if not test else _get_all_files_(hparams.lrs3_synthetic_split, 'val', hparams.lrs3_synthetic_path)
        self.mixing_ratio = hparams.mixing_ratio
        # self.total_size = hparams.total_size
        self.total_size = len(self.lrs3_files)

        random.shuffle(self.lrs3_files)
        random.shuffle(self.lrs3_synthetic_files)

        num_synthetic = int(self.total_size * self.mixing_ratio)
        num_lrs3 = self.total_size - num_synthetic

        selected_lrs3 = self.lrs3_files[:num_lrs3]
        selected_synthetic = self.lrs3_synthetic_files[:num_synthetic]

        self.files = selected_lrs3 + selected_synthetic
        random.shuffle(self.files)

        self.num_frames = hparams.syncnet_T
        self.height = height
        self.width = width
        self.fps = fps
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        while(1):
            index = random.randint(0, len(self.files) - 1)
            fname = self.files[index]
            # if "avi" not in fname:
            #     fname = fname+".avi"
            if "mp4" not in fname:
                fname = fname+".mp4"

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

            out_dict = {
                'video': video,
                'audio': audio,
                'sample': fname
            }

            return out_dict

    def load_frames(self, fname):
        try:
            vr = VideoReader(fname, ctx=cpu(0), width=self.width, height=self.height)
        except:
            return None, None

        try:
            if len(vr)-self.num_frames-1 <= 0:
                return None, None
            
            start_frame = random.choice(range(0, len(vr)-self.num_frames-1))
            end_frame = start_frame + self.num_frames

            selected_frames = range(start_frame, end_frame)
            input_frames = vr.get_batch(selected_frames).asnumpy()
        except:
            return None, None

        input_frames = np.asarray(input_frames) / 255.

        input_frames = np.transpose(input_frames, (3, 0, 1, 2)) 
        
        return input_frames, start_frame

    def load_audio(self, fname, start_frame):
        try:
            # # lrs3 dataset
            wavpath = fname.replace("mp4", "wav")
            wavpath = wavpath.replace(".mp4", ".wav")
            audio = load_wav(wavpath).astype('float32')

            # lrs2 dataset
            # wavpath = fname.replace("vid", "wav/clean")
            # wavpath = wavpath.replace(".mp4", ".wav")
            # audio = load_wav(wavpath).astype('float32')

        except:
            return None, None

        aud_fact = int(np.round(self.sample_rate / self.fps))
        audio_window = audio[aud_fact*start_frame : aud_fact*(start_frame+self.num_frames)]
        audio_window = np.array(audio_window)

        return audio_window, aud_fact


class CMD_Animated_Dataset(data.Dataset):
    def __init__(self, cfg, height=270, width=480, fps=25, sample_rate=16000):
        self.movie_title = cfg.movie_title
        self.data_folder = cfg.data_folder
        self.num_frames = cfg.syncnet_T
        self.height = height
        self.width = width
        self.fps = fps
        self.sample_rate = sample_rate
        self.files = self.get_files()

    def __len__(self):
        return len(self.files)
    
    def get_files(self):
        if self.movie_title is not None:
            file_ls = glob(os.path.join(self.data_folder, self.movie_title) + "/*/*")
        else:
            file_ls = glob(self.data_folder + "/*/*/*")
        random.shuffle(file_ls)
        return file_ls

    def __getitem__(self, index):
        while(1):
            index = random.randint(0, len(self.files) - 1)
            fname = self.files[index]
            if "mp4" not in fname:
                fname = fname+".mp4"

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

            out_dict = {'video': video, 'audio': audio, 'sample': fname}
            return out_dict

    def load_frames(self, fname):
        try:
            vr = VideoReader(fname, ctx=cpu(0), width=self.width, height=self.height)
        except:
            return None, None

        try:
            if len(vr)-self.num_frames-1 <= 0:
                return None, None
            
            start_frame = random.choice(range(0, len(vr)-self.num_frames-1))
            end_frame = start_frame + self.num_frames

            selected_frames = range(start_frame, end_frame)
            input_frames = vr.get_batch(selected_frames).asnumpy()
        except:
            return None, None

        input_frames = np.asarray(input_frames) / 255.

        input_frames = np.transpose(input_frames, (3, 0, 1, 2)) 
        
        return input_frames, start_frame

    def load_audio(self, fname, start_frame):
        try:
            wavpath = fname.replace("/vid/", "/aud/")
            wavpath = wavpath.replace(".mp4", ".wav")
            audio = load_wav(wavpath).astype('float32')
        except:
            return None, None

        aud_fact = int(np.round(self.sample_rate / self.fps))
        audio_window = audio[aud_fact*start_frame : aud_fact*(start_frame+self.num_frames)]
        audio_window = np.array(audio_window)
        return audio_window, aud_fact

class CMD_Animated_DemoDataset(data.Dataset):
    def __init__(self, cfg, height=270, width=480, fps=25, sample_rate=16000):
        self.files = [cfg.video_path]
        self.num_frames = cfg.syncnet_T
        self.height = height
        self.width = width
        self.fps = fps
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        while(1):
            index = random.randint(0, len(self.files) - 1)
            fname = self.files[index]

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

            out_dict = {'video': video, 'audio': audio, 'sample': fname}
            return out_dict

    def load_frames(self, fname):
        try:
            vr = VideoReader(fname, ctx=cpu(0), width=self.width, height=self.height)
        except:
            return None, None

        try:
            if len(vr)-self.num_frames-1 <= 0:
                return None, None
            
            start_frame = random.choice(range(0, len(vr)-self.num_frames-1))
            end_frame = start_frame + self.num_frames

            selected_frames = range(start_frame, end_frame)
            input_frames = vr.get_batch(selected_frames).asnumpy()
        except:
            return None, None

        input_frames = np.asarray(input_frames) / 255.

        input_frames = np.transpose(input_frames, (3, 0, 1, 2)) 
        
        return input_frames, start_frame

    def load_audio(self, fname, start_frame):
        try:
            wavpath = fname.replace("/vid/", "/aud/")
            wavpath = wavpath.replace(".mp4", ".wav")
            audio = load_wav(wavpath).astype('float32')
        except:
            return None, None

        aud_fact = int(np.round(self.sample_rate / self.fps))
        audio_window = audio[aud_fact*start_frame : aud_fact*(start_frame+self.num_frames)]
        audio_window = np.array(audio_window)
        return audio_window, aud_fact


class Simpsons_Dataset(data.Dataset):
    def __init__(self, cfg, height=270, width=480, fps=25, sample_rate=16000, test=False):
        self.data_folder = cfg.data_folder
        self.num_frames = cfg.syncnet_T
        self.height = height
        self.width = width
        self.fps = fps
        self.sample_rate = sample_rate
        self.files = self.get_files(test)

    def __len__(self):
        return len(self.files)
    
    def get_files(self, test):
        file_ls = glob(self.data_folder + "/*/*/*")

        # file_ls_ = []
        # for file in file_ls:
        #     if "season_17" in file or "season_18" in file:
        #         file_ls_.append(file)

        # file_ls = file_ls_

        keep = max(1, int(len(file_ls) * 0.1))
        if test:
            file_ls = file_ls[:keep]
        else:
            file_ls = file_ls[keep:]

            # file_ls_ = []
            # for file in file_ls:
            #     if "season_17" in file:
            #         file_ls_.append(file)

            # file_ls = file_ls_
            # mid = len(file_ls) // 8
            # file_ls = file_ls[:mid]
        random.shuffle(file_ls)
        return file_ls

    def __getitem__(self, index):
        while(1):
            index = random.randint(0, len(self.files) - 1)
            fname = self.files[index]
            if "mp4" not in fname:
                fname = fname+".mp4"

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

            out_dict = {'video': video, 'audio': audio, 'sample': fname}
            return out_dict

    def load_frames(self, fname):
        try:
            vr = VideoReader(fname, ctx=cpu(0), width=self.width, height=self.height)
        except:
            return None, None

        try:
            if len(vr)-self.num_frames-1 <= 0:
                return None, None
            
            start_frame = random.choice(range(0, len(vr)-self.num_frames-1))
            end_frame = start_frame + self.num_frames

            selected_frames = range(start_frame, end_frame)
            input_frames = vr.get_batch(selected_frames).asnumpy()
        except:
            return None, None

        input_frames = np.asarray(input_frames) / 255.

        input_frames = np.transpose(input_frames, (3, 0, 1, 2)) 
        
        return input_frames, start_frame

    def load_audio(self, fname, start_frame):
        try:
            wavpath = fname.replace("/vid/", "/aud/")
            wavpath = wavpath.replace(".mp4", ".wav")
            audio = load_wav(wavpath).astype('float32')
        except:
            return None, None

        aud_fact = int(np.round(self.sample_rate / self.fps))
        audio_window = audio[aud_fact*start_frame : aud_fact*(start_frame+self.num_frames)]
        audio_window = np.array(audio_window)
        return audio_window, aud_fact
