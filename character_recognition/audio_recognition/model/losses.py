"""
Training objective for the audio-visual synchronisation model.

Within a clip, each video chunk has to be matched to the audio chunk it was cut
from: the chunk-by-chunk similarity matrix is used as the logits of a symmetric
contrastive loss (video->audio "row" + audio->video "column"). A second term
pushes the *spatially averaged* similarity to zero, so that a chunk can only
score highly through a localised peak - the mouth - rather than by making the
whole frame agree with the audio.
"""

import torch
from torch import nn
import torch.nn.functional as F

cross_entropy_loss = nn.CrossEntropyLoss()
cos_sim = nn.CosineSimilarity(dim=1)


def accuracy(output, target, topk=(1, 5)):
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:min(k, maxk)].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def sync_contrastive_loss(vid_emb, aud_emb, logits_scale, similarity="max",
                          scale_mean=True):
    """
    :param vid_emb: B x D x n x h x w, one embedding map per video chunk
    :param aud_emb: B x D x n, one embedding per audio chunk
    :param logits_scale: the model's learnt temperature (nn.Linear(1, 1))
    :param similarity: how a chunk pair is scored from its spatial similarity
        map, "max" (peak, i.e. the speaking mouth) or "mean"
    :param scale_mean: map the mean similarity from [-1, 1] to [0, 1] before the
        binary cross entropy, which is only defined on [0, 1]
    :return: dict of loss terms and top-k accuracies, averaged over the batch

    The similarity matrix is built one clip at a time: expanding a clip to all
    chunk pairs already costs O(n^2 * D * h * w), so batching that too would
    multiply the peak memory by the batch size for no gain.
    """
    if similarity not in ("max", "mean"):
        raise ValueError(f"Unknown similarity type {similarity!r}, expected 'max' or 'mean'")

    batch_size = vid_emb.size(0)
    n_chunks = aud_emb.size(2)
    label = torch.arange(n_chunks, device=aud_emb.device)

    losses_row, losses_col, losses_bce = [], [], []
    p1s_row, p5s_row = [], []
    p1s_col, p5s_col = [], []

    for i in range(batch_size):
        ft_v = vid_emb[[i], :, :].transpose(2, 0)
        ft_a = aud_emb[[i], :, :].transpose(2, 0)

        # n_video_chunks x n_audio_chunks x h x w
        spatial_sim = cos_sim(
            ft_v.expand(-1, -1, n_chunks, -1, -1),
            ft_a.expand(-1, -1, n_chunks).transpose(0, 2).unsqueeze(-1).unsqueeze(-1))

        mean_sim = torch.mean(
            spatial_sim.reshape(spatial_sim.shape[0], spatial_sim.shape[1], -1), axis=-1)

        if similarity == "mean":
            output = logits_scale(mean_sim.unsqueeze(-1)).squeeze(-1)
        else:
            max_sim = torch.amax(spatial_sim, dim=(2, 3))
            output = logits_scale(max_sim.unsqueeze(-1)).squeeze(-1)

        # --- suppress the off-peak (spatially averaged) similarity
        target_mean = torch.zeros_like(mean_sim)
        if scale_mean:
            mean_sim = (mean_sim + 1) / 2
        # cosine similarity can land marginally outside [-1, 1] in half precision
        losses_bce.append(
            F.binary_cross_entropy(mean_sim.clamp(0.0, 1.0), target_mean))

        # --- match every video chunk to its audio chunk, and vice versa
        losses_row.append(cross_entropy_loss(output, label))
        p1, p5 = accuracy(output, label)
        p1s_row.append(p1.item())
        p5s_row.append(p5.item())

        losses_col.append(cross_entropy_loss(output.T, label))
        p1, p5 = accuracy(output.T, label)
        p1s_col.append(p1.item())
        p5s_col.append(p5.item())

    def _mean(values):
        return sum(values) / len(values)

    return {
        'loss_row': _mean(losses_row),
        'loss_col': _mean(losses_col),
        'loss_bce': _mean(losses_bce),
        'p1_row': _mean(p1s_row),
        'p5_row': _mean(p5s_row),
        'p1_col': _mean(p1s_col),
        'p5_col': _mean(p5s_col),
    }
