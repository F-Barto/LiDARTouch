import numpy as np
import torch
import torch.nn as nn

from lidartouch.networks.layers.basic_blocks import PaddedConv3x3Block, nearest_upsample, SubPixelUpsamplingBlock
from lidartouch.networks.predictor.utils import create_multiscale_predictor
from functools import partial

class MultiscalePredictionDecoder(nn.Module):
    def __init__(self, chans_enc, activation, scales=4, predictor='inv_depth', upsample_mode='nearest',
                 blur=True, blur_at_end=True, end_skip=True, offsets=1, no_skip=False):
        super(MultiscalePredictionDecoder, self).__init__()

        self.scales = scales
        self.activation = activation
        self.offsets = offsets
        self.end_skip = end_skip

        available_upmodes = ['nearest', 'pixelshuffle', 'res-pixelshuffle']
        if upsample_mode not in available_upmodes:
            raise ValueError(f"upsample_mode must be in {available_upmodes} | upsample_mode={upsample_mode}")
        self.upsample_mode = upsample_mode

        self.chans_enc = chans_enc

        self.nb_stages_enc = len(self.chans_enc)
        self.nb_stages_dec = self.nb_stages_enc - int(end_skip) + offsets

        if self.nb_stages_dec == 5:
            self.chans_dec = np.array([16, 32, 64, 128, 256])
        elif self.nb_stages_dec == 4:
            self.chans_dec = np.array([16, 32, 64, 256])
        else:
            raise ValueError(f"encoder too small: {self.chans_enc}"
                             f" or params to big offsets: {offsets}, end_skip: {end_skip}")

        self.skips = range(offsets, self.nb_stages_enc - int(end_skip) + offsets)
        if no_skip:
            self.skips = []

        # decoder
        self.convs = nn.ModuleDict()
        for i in range(self.nb_stages_dec-1, -1, -1): # [4, 3, 2, 1, 0] for  self.nb_stages_dec=5

            if end_skip or i < (self.nb_stages_dec-1):
                # upconv_0, pre upsampling
                chans_in = self.chans_enc[-1] if i == (self.nb_stages_dec-1) else self.chans_dec[i + 1]
                chans_out = self.chans_dec[i]
                self.convs[f"upconv_{i}_0"] = PaddedConv3x3Block(chans_in, chans_out, self.activation)

                if 'pixelshuffle' in self.upsample_mode:
                    do_blur = blur and (i != 0 or blur_at_end)
                    self.convs[f"pixelshuffle_{i}"] = SubPixelUpsamplingBlock(chans_out, upscale_factor=2, blur=do_blur)

            # upconv_1, post upsampling, post skip or before predictor
            if not end_skip and i == (self.nb_stages_dec-1):
                if no_skip:
                    chans_in =  self.chans_enc[-1]
                else:
                    chans_in =  0
            else:
                chans_in = self.chans_dec[i]

            if i in self.skips:
                chans_in += self.chans_enc[i - offsets]
            chans_out = self.chans_dec[i]
            self.convs[f"upconv_{i}_1"] = PaddedConv3x3Block(chans_in, chans_out, self.activation)

        self.predictor = create_multiscale_predictor(predictor, self.scales, in_chans=self.chans_dec[:self.scales])

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        if self.activation is nn.ReLU:
            initializer = partial(nn.init.kaiming_normal_, mode='fan_out', nonlinearity='relu')
        else:# ELU
            initializer = nn.init.xavier_uniform_

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                initializer(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input_features, **kwargs):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(self.nb_stages_dec-1, -1, -1): # [4, 3, 2, 1, 0] for  self.nb_stages_dec=5

            if self.end_skip or  i < (self.nb_stages_dec-1):
                x = self.convs[f"upconv_{i}_0"](x)

                if self.upsample_mode == 'pixelshuffle':
                    x = self.convs[f"pixelshuffle_{i}"](x)
                if self.upsample_mode == 'res-pixelshuffle':
                    x = self.convs[f"pixelshuffle_{i}"](x) + nearest_upsample(x)
                if self.upsample_mode == 'nearest':
                    x = nearest_upsample(x)

                if i in self.skips:
                    x = [x, input_features[i - self.offsets]]
                    x = torch.cat(x, 1)

            x = self.convs[f"upconv_{i}_1"](x)

            if i in range(self.scales):
                self.predictor(x, i, **kwargs)

        return self.predictor.compile_predictions()