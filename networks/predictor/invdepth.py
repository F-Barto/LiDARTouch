from functools import partial

import torch
import torch.nn as nn

from networks.common.basic_blocks import PaddedConv3x3, disp_to_depth
from networks.predictor.base import MultiScaleBasePredictor


class InvDepthPredictor(nn.Module):
    def __init__(self, in_chans, prefix='', postfix='', min_depth=0.1, max_depth=120.0):
        super(InvDepthPredictor, self).__init__()
        self.invdepthconv = PaddedConv3x3(in_chans, 1)
        self.sigmoid = nn.Sigmoid()

        self.prefix = prefix
        self.postfix = postfix

        self.scale_inv_depth = partial(disp_to_depth, min_depth=min_depth, max_depth=max_depth)

    def forward(self, x):
        output = self.sigmoid(self.invdepthconv(x))
        scaled_value = self.scale_inv_depth(output)[0]

        return {self.prefix + f'inv_depths' + self.postfix: scaled_value}

class MultiScaleInvDepthPredictor(MultiScaleBasePredictor):
    def __init__(self, scales, in_chans, prefix='', postfix='', min_depth=0.1, max_depth=120.0):
        super(MultiScaleInvDepthPredictor, self).__init__(scales)

        assert len(in_chans) == scales
        self.n = scales

        self.sigmoid = nn.Sigmoid()

        self.prefix = prefix
        self.postfix = postfix

        self.invdepthconvs = nn.ModuleDict([
            (f'invdepthconv_{i}', PaddedConv3x3(in_chans[i], 1)) for i in range(self.n)
        ])

        self.outputs = {i:None for i in range(self.n)}

        self.scale_inv_depth = partial(disp_to_depth, min_depth=min_depth, max_depth=max_depth)

    def forward(self, x, i, **kwargs):
        if i >= self.scales: raise IndexError(f'The network has at most {self.scales} of prediction.')

        output = self.sigmoid(self.invdepthconvs[f'invdepthconv_{i}'](x))

        self.outputs[i] = output

    def get_prediction(self, i):
        if self.outputs[i] is  None:
            raise ValueError(f'Prediction of scale {i} not yet computed')

        return self.scale_inv_depth(self.outputs[i])[0]

    def compile_predictions(self):

        scaled_values = [self.get_prediction(i) for i in range(self.n)]

        if self.training:
            return {self.prefix+f'inv_depths'+self.postfix: scaled_values}
        else:
            return {self.prefix+f'inv_depths'+self.postfix: self.get_prediction(0)}


class ScaledMultiScaleInvDepthPredictor(MultiScaleInvDepthPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #self.normalizer = nn.InstanceNorm2d(1)

    def forward(self, x, i, gammas=None, betas=None):
        if i >= self.scales: raise IndexError(f'The network has at most {self.scales} of prediction.')

        output = self.invdepthconvs[f'invdepthconv_{i}'](x)

        #normalized_output = output / torch.mean(output, dim=[2,3], keepdim=True)
        #normalized_output = self.normalizer(output)
        normalized_output = output / output.norm(dim=[2,3], keepdim=True)

        assert gammas is not None and betas is not None, "Check your config"
        scaled_output = torch.mul(normalized_output, gammas[i]) + betas[i]

        self.outputs[i] = self.sigmoid(scaled_output)