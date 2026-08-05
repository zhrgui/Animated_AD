"""
LWTNet: the two-stream audio-visual synchronisation network from
"Self-Supervised Learning of Audio-Visual Objects from Video" (Afouras et al.),
plus the temporal-adapter variant that we fine-tune on animated content.
"""

import torch
import torch.nn as nn

from utils.net import DebugModule, calc_receptive_field


class LWTNet(nn.Module):
    """
    Video and audio encoders projecting into a shared 1024-d embedding space.

    forward_vid: B x 3 x t x H x W  ->  B x 1024 x t' x h x w
    forward_aud: B x 1 x F x m      ->  B x 1024 x m'
    """

    def __init__(self):
        super().__init__()

        self.net_lip = self.build_net_vid()
        self.ff_lip = NetFC(input_dim=512, hidden_dim=512, embed_dim=1024)

        self.net_aud = self.build_net_aud()
        self.ff_aud = NetFC(input_dim=512, hidden_dim=512, embed_dim=1024)

        _, _, _, self.start_offset = calc_receptive_field(self.net_lip.layers,
                                                          imsize=400)

        self.logits_scale = nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.logits_scale.weight)

    def build_net_vid(self):
        layers = [
            {
                'type': 'conv3d',
                'n_channels': 64,
                'kernel_size': (5, 7, 7),
                'stride': (1, 2, 2),
                'padding': (0),
                'maxpool': {
                    'kernel_size': (1, 3, 3),
                    'stride': (1, 2, 2)
                }
            },
            {
                'type': 'conv3d',
                'n_channels': 128,
                'kernel_size': (1, 5, 5),
                'stride': (1, 2, 2),
                'padding': (0, 0, 0),
            },
            {
                'type': 'conv3d',
                'n_channels': 256,
                'kernel_size': (1, 3, 3),
                'stride': (1, 1, 1),
                'padding': (0, 1, 1),
            },
            {
                'type': 'conv3d',
                'n_channels': 256,
                'kernel_size': (1, 3, 3),
                'stride': (1, 1, 1),
                'padding': (0, 1, 1),
            },
            {
                'type': 'conv3d',
                'n_channels': 256,
                'kernel_size': (1, 3, 3),
                'stride': (1, 1, 1),
                'padding': (0, 1, 1),
                'maxpool': {
                    'kernel_size': (1, 3, 3),
                    'stride': (1, 2, 2)
                }
            },
            {
                'type': 'fc3d',
                'n_channels': 512,
                'kernel_size': (1, 5, 5),
                'stride': (1, 1, 1),
                'padding': (0),
            },
        ]
        return VGGNet(n_channels_in=3, layers=layers)

    def build_net_aud(self):
        layers = [
            {
                'type': 'conv2d',
                'n_channels': 64,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 1),
                'maxpool': {
                    'kernel_size': (3, 1),
                    'stride': (2, 1)
                }
            },
            {
                'type': 'conv2d',
                'n_channels': 192,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
                'maxpool': {
                    'kernel_size': (3, 3),
                    'stride': (2, 2)
                }
            },
            {
                'type': 'conv2d',
                'n_channels': 384,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
            },
            {
                'type': 'conv2d',
                'n_channels': 256,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
            },
            {
                'type': 'conv2d',
                'n_channels': 256,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (1, 1),
                'maxpool': {
                    'kernel_size': (2, 3),
                    'stride': (2, 2)
                }
            },
            {
                'type': 'fc2d',
                'n_channels': 512,
                'kernel_size': (4, 4),
                'stride': (1, 1),
                'padding': (0, 0),
            },
        ]
        return VGGNet(n_channels_in=1, layers=layers)

    def forward_vid(self, x, return_feats=False):
        out_conv6 = self.net_lip(x)
        out = self.ff_lip(out_conv6)
        if return_feats:
            return out, out_conv6
        else:
            return out

    def forward_aud(self, x):
        out = self.net_aud(x)
        if len(out.shape) < 5:
            out = out[..., None]
        out = self.ff_aud(out)
        # squeeze the spatial dimensions of audio - those will always be constant
        out = out.squeeze(-1).squeeze(-2)
        return out

    def forward(self, video, audio, return_feats=False):
        """
        Encode a video and an audio batch in one call.

        Training goes through this (rather than calling forward_vid / forward_aud
        directly) so that DistributedDataParallel sees a single forward pass and
        actually synchronises the gradients across ranks.
        """
        vid_emb, vid_feats = self.forward_vid(video, return_feats=True)
        aud_emb = self.forward_aud(audio)
        if return_feats:
            return vid_emb, aud_emb, vid_feats
        return vid_emb, aud_emb


class LWTNet_temporal_adapter(LWTNet):
    """
    LWTNet with a temporal adapter on top of each embedding head.

    Both adapters collapse the 11-step receptive field of a 15-frame / 60-mel
    chunk into a single embedding, so `forward_vid` returns a singleton temporal
    dimension. This is the variant that gets fine-tuned on animated video; the
    encoders are initialised from a pretrained LWTNet checkpoint and the adapters
    start from scratch (see `model.checkpoint.load_model_params`).
    """

    def __init__(self):
        super().__init__()

        self.vid_adapter = nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=(11, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.aud_adapter = nn.Conv3d(in_channels=1024, out_channels=1024, kernel_size=(1, 11, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward_vid(self, x, return_feats=False):
        out_conv6 = self.net_lip(x)
        out = self.ff_lip(out_conv6)
        out = self.vid_adapter(out)
        if return_feats:
            return out, out_conv6
        else:
            return out

    def forward_aud(self, x):
        out = self.net_aud(x)
        if len(out.shape) < 5:
            out = out[..., None]
        out = self.ff_aud(out)
        out = self.aud_adapter(out)
        # squeeze the spatial dimensions of audio - those will always be constant
        out = out.squeeze(-1).squeeze(-2)
        return out


class VGGNet(DebugModule):

    conv_dict = {
        'conv1d': nn.Conv1d,
        'conv2d': nn.Conv2d,
        'conv3d': nn.Conv3d,
        'fc1d': nn.Conv1d,
        'fc2d': nn.Conv2d,
        'fc3d': nn.Conv3d,
    }

    pool_dict = {
        'conv1d': nn.MaxPool1d,
        'conv2d': nn.MaxPool2d,
        'conv3d': nn.MaxPool3d,
    }

    norm_dict = {
        'conv1d': nn.BatchNorm1d,
        'conv2d': nn.BatchNorm2d,
        'conv3d': nn.BatchNorm3d,
        'fc1d': nn.BatchNorm1d,
        'fc2d': nn.BatchNorm2d,
        'fc3d': nn.BatchNorm3d,
    }

    def __init__(self, n_channels_in, layers):
        super(VGGNet, self).__init__()

        self.layers = layers

        n_channels_prev = n_channels_in
        for l_id, lr in enumerate(self.layers):
            l_id += 1
            name = 'fc' if 'fc' in lr['type'] else 'conv'
            conv_type = self.conv_dict[lr['type']]
            norm_type = self.norm_dict[lr['type']]
            self.__setattr__(
                '{:s}{:d}'.format(name, l_id),
                conv_type(n_channels_prev,
                          lr['n_channels'],
                          kernel_size=lr['kernel_size'],
                          stride=lr['stride'],
                          padding=lr['padding']))
            n_channels_prev = lr['n_channels']
            self.__setattr__('bn{:d}'.format(l_id), norm_type(lr['n_channels']))
            if 'maxpool' in lr:
                pool_type = self.pool_dict[lr['type']]
                padding = lr['maxpool']['padding'] if 'padding' in lr[
                    'maxpool'] else 0
                self.__setattr__(
                    'mp{:d}'.format(l_id),
                    pool_type(kernel_size=lr['maxpool']['kernel_size'],
                              stride=lr['maxpool']['stride'],
                              padding=padding),
                )

    def forward(self, inp):
        self.debug_line('Input', inp)
        out = inp
        for l_id, lr in enumerate(self.layers):
            l_id += 1
            name = 'fc' if 'fc' in lr['type'] else 'conv'
            out = self.__getattr__('{:s}{:d}'.format(name, l_id))(out)
            out = self.__getattr__('bn{:d}'.format(l_id))(out)
            out = nn.ReLU(inplace=True)(out)
            self.debug_line(self.__getattr__('{:s}{:d}'.format(name, l_id)),
                            out)
            if 'maxpool' in lr:
                out = self.__getattr__('mp{:d}'.format(l_id))(out)
                self.debug_line(self.__getattr__('mp{:d}'.format(l_id)), out)

        self.debug_line('Output', out, final_call=True)

        return out


class NetFC(DebugModule):

    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(NetFC, self).__init__()
        self.fc7 = nn.Conv3d(input_dim, hidden_dim, kernel_size=(1, 1, 1))
        self.bn7 = nn.BatchNorm3d(hidden_dim)
        self.fc8 = nn.Conv3d(hidden_dim, embed_dim, kernel_size=(1, 1, 1))

    def forward(self, inp):
        out = self.fc7(inp)
        self.debug_line(self.fc7, out)
        out = self.bn7(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.fc8(out)
        self.debug_line(self.fc8, out, final_call=True)
        return out
