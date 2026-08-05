import math

import torch
from torch import nn
from tqdm import tqdm


class DebugModule(nn.Module):
    """
    Wrapper class for printing the activation dimensions
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.debug_log = False

    def debug_line(self, layer_str, output, memuse=1, final_call=False):
        if self.debug_log:
            namestr = '{}: '.format(self.name) if self.name is not None else ''
            print('{}{:80s}: dims {}'.format(namestr, repr(layer_str),
                                             output.shape))

            if final_call:
                self.debug_log = False
                print()


def calc_receptive_field(layers, imsize, layer_names=None):
    if layer_names is not None:
        print("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]

    for l_id, layer in enumerate(layers):
        conv = [
            layer[key][-1] if type(layer[key]) in [list, tuple] else layer[key]
            for key in ['kernel_size', 'stride', 'padding']
        ]
        currentLayer = outFromIn(conv, currentLayer)
        if 'maxpool' in layer:
            conv = [
                (layer['maxpool'][key][-1] if type(layer['maxpool'][key])
                 in [list, tuple] else layer['maxpool'][key]) if
                (not key == 'padding' or 'padding' in layer['maxpool']) else 0
                for key in ['kernel_size', 'stride', 'padding']
            ]
            currentLayer = outFromIn(conv, currentLayer, ceil_mode=False)
    return currentLayer


def outFromIn(conv, layerIn, ceil_mode=True):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


def logsoftmax_2d(logits):
    # Log softmax on last 2 dims because torch won't allow multiple dims
    orig_shape = logits.shape
    logprobs = torch.nn.LogSoftmax(dim=-1)(
        logits.reshape(list(logits.shape[:-2]) + [-1])).reshape(orig_shape)
    return logprobs


def run_func_in_parts(func, vid_emb, aud_emb, part_len, dim, device):
    """
    Run given function in parts, spliting the inputs on dimension dim
    This is used to save memory when inputs too large to compute on gpu
    """
    dist_chunk = []
    for v_spl, a_spl in tqdm(list(
            zip(vid_emb.split(part_len, dim=dim),
                aud_emb.split(part_len, dim=dim))),
                             desc='Calculating pairwise scores'):
        dist_chunk.append(func(v_spl.to(device), a_spl.to(device)))
    dist = torch.cat(dist_chunk, dim - 1)
    return dist
