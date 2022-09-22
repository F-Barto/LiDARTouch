from functools import partial

import torch
import torch.nn as nn

from lidartouch.networks.layers.basic_blocks import PaddedConv3x3, scaled_disp
from lidartouch.networks.predictor.base import MultiScaleBasePredictor


class InvDepthPredictor(nn.Module):
    def __init__(self, in_chans, prefix='', postfix='', min_depth=0.5, max_depth=100.):
        super(InvDepthPredictor, self).__init__()
        self.invdepthconv = PaddedConv3x3(in_chans, 1)
        self.sigmoid = nn.Sigmoid()

        self.prefix = prefix
        self.postfix = postfix
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x):
        out = self.sigmoid(self.invdepthconv(x))
        out = scaled_disp(out, self.min_depth, self.max_depth)
        #out = self.sigmoid(self.invdepthconv(x)) / self.min_depth

        return {self.prefix + f'inv_depths' + self.postfix: out}

class MultiScaleInvDepthPredictor(MultiScaleBasePredictor):
    def __init__(self, scales, in_chans, prefix='', postfix='', min_depth=0.5, max_depth=100.):
        super(MultiScaleInvDepthPredictor, self).__init__(scales)

        assert len(in_chans) == scales
        self.n = scales

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.sigmoid = nn.Sigmoid()

        self.prefix = prefix
        self.postfix = postfix

        self.invdepthconvs = nn.ModuleDict([
            (f'invdepthconv_{i}', PaddedConv3x3(in_chans[i], 1)) for i in range(self.n)
        ])

        self.outputs = {i:None for i in range(self.n)}

    def forward(self, x, i, **kwargs):
        if i >= self.scales:
            raise IndexError(f'The network has at most {self.scales} of prediction.')

        out = self.sigmoid(self.invdepthconvs[f'invdepthconv_{i}'](x))
        out = scaled_disp(out, self.min_depth, self.max_depth)
        #out = self.sigmoid(self.invdepthconvs[f'invdepthconv_{i}'](x)) / self.min_depth

        self.outputs[i] = out

    def get_prediction(self, i):
        if self.outputs[i] is None:
            raise ValueError(f'Prediction of scale {i} not yet computed')

        return self.outputs[i]

    def compile_predictions(self):

        predictions = [self.get_prediction(i) for i in range(self.n)]

        if self.training:
            return {self.prefix+f'inv_depths'+self.postfix: predictions}
        else:
            return {self.prefix+f'inv_depths'+self.postfix: self.get_prediction(0)}