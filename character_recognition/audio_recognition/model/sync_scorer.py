"""
Inference-time wrappers around LWTNet.

`SyncScorerBase` holds everything that is shared: it slides the model over a
clip, aligns the two streams by the offset the model itself considers optimal,
and exposes the audio-visual similarity. The two public classes differ only in
what they do with the result - a per-frame score (used by the visual enhancement
of audio character recognition) or an attention-map video (used for inspection).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.audio import wav2filterbanks
from utils.net import logsoftmax_2d, run_func_in_parts
from utils.viz import VideoSaver, viz_sync_attn


class SyncScorerBase:

    def __init__(self, model, opts):
        self.model = model
        self.opts = opts

        self.device = torch.device(f'cuda:{opts.gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.step = 0

    # ------------------------------------------------------------------ encoding

    def encode(self, video, audio, progress=False):
        """
        Run the encoders over a whole clip with a 15-frame / 60-mel sliding window.

        :param video: 1 x c x T x H x W, in 0..255 - the range the model is fed,
            here and in training (`data.datasets.AVWindowDataset`)
        :param audio: 1 x T*sample_rate/fps
        :return: (video, vid_emb, aud_emb) with the video padded by the model's
            receptive field offset, ready for `online_sync`
        """
        mel, _, _, _ = wav2filterbanks(audio.to(self.device))

        pad_len = int(self.model.start_offset)
        video = torch.nn.ConstantPad3d([pad_len, pad_len, pad_len, pad_len], 0)(video)
        input_frames = np.asarray(video)

        input_frames = np.array([
            input_frames[:, :, i:i + 15, ]
            for i in range(0, input_frames.shape[2], 1)
            if (i + 15 <= input_frames.shape[2])
        ])
        num_frames = input_frames.shape[0]

        frames = np.transpose(input_frames, (1, 2, 0, 3, 4, 5))
        vid_chunk = torch.FloatTensor(np.array(frames))

        spec = mel.squeeze(0).cpu().numpy()
        spec = np.array([
            spec[i:i + 60, :] for i in range(0, spec.shape[0], 4)
            if (i + 60 <= spec.shape[0])
        ])
        spec = spec[:num_frames]
        mel_chunk = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).permute(0, 1, 2, 4, 3)

        B = vid_chunk.size(0)
        face_sequences = torch.cat([vid_chunk[:, :, i] for i in range(vid_chunk.size(2))], dim=0)
        audio_sequences = torch.cat([mel_chunk[:, :, i] for i in range(mel_chunk.size(2))], dim=0)

        batch_size = 4
        vid_emb, aud_emb = [], []

        window_iter = range(0, len(face_sequences), batch_size)
        if progress:
            window_iter = tqdm(window_iter)

        for i in window_iter:
            video_inp = face_sequences[i:i + batch_size, ]
            audio_inp = audio_sequences[i:i + batch_size, ]

            ve = self.model.forward_vid(video_inp.to(self.device))
            ae = self.model.forward_aud(audio_inp.to(self.device))

            vid_emb.append(ve.detach().cpu())
            aud_emb.append(ae.detach().cpu())

            torch.cuda.empty_cache()

        aud_emb = torch.cat(aud_emb, dim=0)
        vid_emb = torch.cat(vid_emb, dim=0)

        vid_emb = torch.nn.functional.normalize(vid_emb, p=2, dim=1)
        aud_emb = torch.nn.functional.normalize(aud_emb, p=2, dim=1)

        aud_emb = torch.stack(torch.split(aud_emb, B, dim=0), dim=2).squeeze(3)
        vid_emb = torch.stack(torch.split(vid_emb, B, dim=0), dim=2).squeeze(3)

        aud_emb = aud_emb[:, None]

        return video, vid_emb, aud_emb

    # ------------------------------------------------------------- scoring / sync

    def calc_av_scores(self, vid_emb, aud_emb):
        scores = self.calc_att_map(vid_emb, aud_emb)
        att_map = logsoftmax_2d(scores)

        scores = torch.nn.MaxPool3d(kernel_size=(1, scores.shape[-2], scores.shape[-1]))(scores)
        scores = scores.squeeze(-1).squeeze(-1).mean(2)
        return scores, att_map

    def calc_att_map(self, vid_emb, aud_emb):
        vid_emb = vid_emb[:, :, None]
        aud_emb = aud_emb.transpose(1, 2)[..., None, None]

        scores = run_func_in_parts(lambda x, y: (x * y).sum(1), vid_emb, aud_emb, part_len=10, dim=3, device=self.device)
        scores = self.model.logits_scale(scores[..., None]).squeeze(-1)
        return scores

    def online_sync(self, vid_emb, aud_emb, video, audio):
        sync_offset = self.calc_optimal_av_offset(vid_emb, aud_emb)

        vid_emb_sync, aud_emb_sync = self.sync_av_with_offset(vid_emb, aud_emb, sync_offset, dim_v=2, dim_a=3)

        # e.g. a_mult = 640 for 16khz and 25 fps
        a_mult = int(self.opts.sample_rate / self.opts.fps)
        video, audio = self.sync_av_with_offset(video, audio, sync_offset, dim_v=2, dim_a=1, a_mult=a_mult)
        return video, audio, vid_emb_sync, aud_emb_sync

    def create_online_sync_negatives(self, vid_emb, aud_emb):
        assert self.opts.n_negative_samples % 2 == 0
        ww = self.opts.n_negative_samples // 2

        fr_trunc, to_trunc = ww, aud_emb.shape[-1] - ww
        vid_emb_pos = vid_emb[:, :, fr_trunc:to_trunc]
        slice_size = to_trunc - fr_trunc

        aud_emb_posneg = aud_emb.squeeze(1).unfold(-1, slice_size, 1)
        aud_emb_posneg = aud_emb_posneg.permute([0, 2, 1, 3])

        # this is the index of the positive samples within the posneg bundle
        pos_idx = self.opts.n_negative_samples // 2
        aud_emb_pos = aud_emb[:, 0, :, fr_trunc:to_trunc]

        # make sure that we got the indices correctly
        assert torch.all(aud_emb_posneg[:, pos_idx] == aud_emb_pos)
        return vid_emb_pos, aud_emb_posneg, pos_idx

    def calc_optimal_av_offset(self, vid_emb, aud_emb):
        vid_emb, aud_emb, pos_idx = self.create_online_sync_negatives(vid_emb, aud_emb)
        scores, _ = self.calc_av_scores(vid_emb, aud_emb)
        offset = scores.argmax() - pos_idx
        return offset.item()

    def sync_av_with_offset(self, vid_emb, aud_emb, offset, dim_v, dim_a, a_mult=1):
        if vid_emb is not None:
            init_dim = vid_emb.shape[dim_v]
        else:
            init_dim = aud_emb.shape[dim_a] // a_mult

        length = init_dim - int(np.abs(offset))

        if vid_emb is not None:
            if offset < 0:
                vid_emb = vid_emb.narrow(dim_v, -offset, length)
            else:
                vid_emb = vid_emb.narrow(dim_v, 0, length)

            assert vid_emb.shape[dim_v] == init_dim - np.abs(offset)

        if aud_emb is not None:
            if offset < 0:
                aud_emb = aud_emb.narrow(dim_a, 0, length * a_mult)
            else:
                aud_emb = aud_emb.narrow(dim_a, offset * a_mult,
                                         length * a_mult)
            assert aud_emb.shape[dim_a] // a_mult == init_dim - np.abs(offset)
        return vid_emb, aud_emb


class DemoEvalTrainerScoreOnly(SyncScorerBase):
    """Per-frame audio-visual synchronisation score for a single cropped track."""

    def eval(self, dataloader):
        for batch_sample in dataloader:
            self.model.zero_grad()
            video = batch_sample['video']
            audio = batch_sample['audio']

            video, vid_emb, aud_emb = self.encode(video, audio)
            video, audio, vid_emb, aud_emb = self.online_sync(vid_emb, aud_emb, video, audio)

            vid_emb = vid_emb.permute(0, 2, 1, 3, 4)
            aud_emb = aud_emb.squeeze(1).permute(0, 2, 1)
            aud_emb_expanded = aud_emb.unsqueeze(-1).unsqueeze(-1)

            similarity = F.cosine_similarity(aud_emb_expanded, vid_emb, dim=2)
            similarity = (similarity + 1) / 2
            scores = similarity.amax(dim=(-2, -1))
            return scores


class DemoEvalTrainer(SyncScorerBase):
    """Render the audio-visual attention map of a clip as an mp4."""

    def __init__(self, model, opts):
        super().__init__(model, opts)

        opts.output_dir = os.path.join(opts.output_dir)
        self.checkpoints_path = opts.output_dir + "/checkpoints"

        os.makedirs(opts.output_dir, exist_ok=True)
        self.video_saver = VideoSaver(opts.output_dir)

    def eval(self, dataloader):
        bs = self.opts.batch_size

        for batch_sample in dataloader:
            self.model.zero_grad()
            video = batch_sample['video']
            audio = batch_sample['audio']

            video, vid_emb, aud_emb = self.encode(video, audio, progress=True)
            video, audio, vid_emb, aud_emb = self.online_sync(vid_emb, aud_emb, video, audio)

            att_map = self.calc_att_map(vid_emb, aud_emb)
            att_map = logsoftmax_2d(att_map)
            att_map_smooth = self.smoothen_and_pad_att_map(att_map[:, 0])

            if self.opts.batch_size == 1:
                att_map_smooth = att_map_smooth[None]

            audio = audio.reshape((bs, -1) + audio.shape[1:])[:, -1]

            b_id = 0
            viz_sync_attn(video[b_id], audio[b_id], att_map=att_map_smooth[b_id],
                          model_start_offset=int(self.model.start_offset),
                          video_saver=self.video_saver, step=self.step)

            self.step += 1

    def smoothen_and_pad_att_map(self, av_att, pad_resid=7):
        map_t_avg = torch.nn.AvgPool3d(kernel_size=(5, 1, 3), stride=1, padding=(2, 0, 1), count_include_pad=False)(av_att)
        map_t_avg = torch.nn.ReplicationPad3d((0, 0, 0, 0, pad_resid, pad_resid))(map_t_avg[None]).squeeze()
        return map_t_avg
