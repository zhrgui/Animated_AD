"""
Visualisation of the audio-visual attention map produced by the synchronisation
model, used by `model.sync_scorer.DemoEvalTrainer`.

The AVObjects visualisations that depended on the external `aolib_p3` package
(box/trajectory overlays and source separation) were dropped when this folder
was reorganised; only the attention-map video is kept, which is what the
character recognition pipeline uses. See the git history for the originals.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image


class VideoSaver:

    def __init__(self, savedir):
        os.makedirs(savedir, exist_ok=True)
        self.savedir = savedir
        self.id = 0

    def save_mp4_from_vid_and_audio(self,
                                    video_tensor,
                                    audio_wav=None,
                                    fps=25,
                                    sr=16000,
                                    outname=None,
                                    extract_frames_hop=None):
        """
        :param video_tensor: tchw
        :param sr:
        :return:
        """

        from moviepy.video.VideoClip import VideoClip

        video_tensor = video_tensor.transpose([0, 2, 3, 1])  # thwc
        # that's to avoid error due to float precision
        vid_dur = len(video_tensor) * (1. / fps) - 1e-6
        v_clip = VideoClip(lambda t: video_tensor[int(np.round(t * fps))],
                           duration=vid_dur)

        if outname:
            outfile = os.path.join(self.savedir, outname)
            if not outfile.endswith('.mp4'):
                outfile += '.mp4'
        else:
            outfile = os.path.join(self.savedir, '%03d.mp4' % self.id)

        if audio_wav is not None:
            _, temp_audiofile = tempfile.mkstemp(dir='/dev/shm', suffix='.wav')
            import torch
            if isinstance(audio_wav, torch.Tensor):
                audio_wav = audio_wav.numpy()

            import scipy.io.wavfile
            scipy.io.wavfile.write(temp_audiofile, sr, audio_wav)

        self.id += 1
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        _, temp_videofile = tempfile.mkstemp(dir='/dev/shm', suffix='.mp4')

        v_clip.write_videofile(temp_videofile, fps=fps, verbose=False)

        if audio_wav is not None:
            command = [
                "ffmpeg", "-threads", "1", "-loglevel", "error", "-y",
                "-i", temp_videofile, "-i", temp_audiofile,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-pix_fmt", "yuv420p", "-shortest", outfile,
            ]
            subprocess.call(command)
        else:
            shutil.move(temp_videofile, outfile)

        v_clip.close()

        if extract_frames_hop:  # extract the video as frames for paper
            frames_dir = os.path.join(
                os.path.dirname(outfile),
                'frames_' + os.path.basename(outfile).replace('.mp4', ''))
            os.makedirs(frames_dir, exist_ok=True)
            for fr_id, frame in enumerate(video_tensor[::extract_frames_hop]):
                Image.fromarray(frame[:, :-5, :]).save(
                    os.path.join(frames_dir, '%04d.png' % fr_id))


def viz_sync_attn(video,
                  audio,
                  att_map,
                  model_start_offset,
                  video_saver,
                  step,
                  vids_name='avsync_viz'):
    """
    video: c T H W
    att_map: t h w
    """
    print('Vizualizaing av att and avobject trajectories')

    video = video.permute([1, 2, 3, 0]).numpy().astype('uint8')  # C T H W -> T H W C

    # ----------- make cam_vid showing AV-att map and peaks  ---------------
    vid_with_cam = show_cam_on_vid(video,
                                   att_map.detach().cpu(),
                                   offset=model_start_offset)

    # remove padding equal to the model's conv offset
    pad_len = model_start_offset
    vid_with_cam = vid_with_cam[..., pad_len:-pad_len, pad_len:-pad_len]

    video_saver.save_mp4_from_vid_and_audio(
        vid_with_cam,
        audio / 32768,
        outname='{}/{}'.format(vids_name, step),
    )
    print("Saved video at: {}/{}".format(vids_name, step))


def show_cam_on_vid(vid, cam, offset=0):
    """
    :param vid: t x h x w x c
    :param cam: h_att x w_att
    :return:
    """
    assert len(cam) == len(vid), \
        'Attention map and video have different lengths ({} vs {})'.format(len(cam), len(vid))

    vmin = vmax = None
    vid_with_cam = np.array([
        show_cam_on_image(frame, msk, offset, vmin, vmax)
        for frame, msk in zip(vid, cam)
    ])
    return vid_with_cam


def show_cam_on_image(frame, cam, offset, vmin=None, vmax=None):
    """
    :param frame: c x h x w
    :param cam: h_att x w_att
    :return:
    """
    import cv2

    frame = np.float32(frame) / 255

    if vmin is not None:
        vmax = -vmin
        vmin = -vmax

    cam = normalize_img(-cam, vmin=vmin, vmax=vmax)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    h_frame, w_frame = frame.shape[:2]
    heatmap, offset = map_to_full(heatmap,
                                  w_frame,
                                  h_frame,
                                  offset,
                                  w_map=heatmap.shape[1])
    heatmap = np.float32(heatmap) / 255
    heatmap_frame = np.zeros_like(frame)
    heatmap_frame[offset:h_frame - offset, offset:w_frame - offset] = heatmap

    cam = heatmap_frame + frame
    cam = cam / np.max(cam)

    new_img = np.uint8(255 * cam)
    new_img = new_img.transpose([2, 0, 1])  # hwc --> chw
    return new_img


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


# ---- mapping from attention map coordinates back to original image coordinates ----

def map_to_full(map_in, w_frame, h_frame, offset, w_map=None):
    w_map = w_map or map_in.shape[-1]
    hm_im = Image.fromarray(map_in)
    offset, h_map, w_map, h_att, w_att = calc_map_offset(
        offset, h_frame, w_frame, w_map)
    hm_im = hm_im.resize((w_map, h_map))
    map_full = np.array(hm_im)
    return map_full, offset


def calc_map_offset(offset, h_frame, w_frame, w_map):
    # this is without the edge for going between map coords and original image pixels
    w_att, h_att = w_frame - 2 * offset, h_frame - 2 * offset
    edge = int(np.round((w_frame - 2 * offset) / (w_map - 1) / 2))
    offset -= edge
    w_map, h_map = w_frame - 2 * offset, h_frame - 2 * offset
    return offset, h_map, w_map, h_att, w_att
