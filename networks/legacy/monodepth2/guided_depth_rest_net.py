import torch.nn as nn

from functools import partial

from networks.legacy.monodepth2.layers.resnet_encoder import ResnetEncoder
from networks.legacy.monodepth2.layers.depth_decoder import DepthDecoder
from networks.legacy.monodepth2.layers.common import disp_to_depth, get_activation
from networks.legacy.attention_guidance import AttentionGuidance
from networks.legacy.continuous_convolution import ContinuousFusion

from utils.depth import depth2inv

########################################################################################################################


class GuidedDepthResNet(nn.Module):
    """
    Inverse depth network based on the ResNet architecture.
    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, num_layers=18, input_channels=3, activation='relu', guidance='attention', attention_scheme='res-sig',
                 inverse_lidar_input=True, preact=False, invertible=False, n_power_iterations=5, no_maxpool=False,
                 **kwargs):
        super().__init__()

        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        assert guidance in ['attention', 'continuous']

        self.inverse_lidar_input = inverse_lidar_input

        activation_cls = get_activation(activation)

        # keeping the name `encoder` so that we can use pre-trained weight directly
        self.encoder = ResnetEncoder(num_layers=num_layers, input_channels=input_channels, activation=activation_cls,
                                     preact=preact, invertible=invertible, n_power_iterations=n_power_iterations)
        self.lidar_encoder = ResnetEncoder(num_layers=num_layers, input_channels=1, activation=activation_cls,
                                           no_first_norm=True, no_maxpool=no_maxpool, preact=preact, invertible=invertible,
                                           n_power_iterations=n_power_iterations)

        self.num_ch_enc = self.encoder.num_ch_enc
        skip_features_factor = 2 if ('concat' in attention_scheme) or (guidance == 'continuous') else 1
        self.num_ch_skips = [skip_features_factor * num_ch for num_ch in self.num_ch_enc]

        # at each resblock fuse with guidance the features of both encoders
        self.guidances = nn.ModuleDict()
        for i in range(len(self.num_ch_enc)):

            num_ch =  self.num_ch_enc[i]

            if guidance == 'attention':
                self.guidances.update({f"guidance_{i}": AttentionGuidance(num_ch, activation_cls, attention_scheme)})
            elif guidance == 'continuous':
                self.guidances.update({f"guidance_{i}": ContinuousFusion(num_ch * 2, activation_cls)})
            else:
                print(f"guidance {guidance} not implemented")

        self.decoder = DepthDecoder(num_ch_enc=self.num_ch_skips, activation=activation_cls, **kwargs)

        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=120.0)

    def forward(self, cam_input, lidar_input, nn_diff_pts_3d=None, pixel_idxs=None, nn_pixel_idxs=None):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """

        if self.inverse_lidar_input:
            lidar_input = depth2inv(lidar_input)

        cam_features = self.encoder(cam_input)
        lidar_features = self.lidar_encoder(lidar_input)

        self.guided_features = []
        for i in range(len(self.num_ch_enc)):

            if nn_diff_pts_3d is not None: # continuous guidance
                guided_feature = self.guidances[f"guidance_{i}"](cam_features[i], lidar_features[i],
                                                                 nn_diff_pts_3d[i], pixel_idxs[i], nn_pixel_idxs[i])
            else:
                guided_feature = self.guidances[f"guidance_{i}"](cam_features[i], lidar_features[i])

            self.guided_features.append(guided_feature)

        x = self.decoder(self.guided_features)

        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]