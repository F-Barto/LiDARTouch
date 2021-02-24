import torch.nn as nn
import torch

from functools import partial

from networks.legacy.monodepth2.layers.resnet_encoder import ResnetEncoder
from networks.legacy.monodepth2.layers.depth_decoder import DepthDecoder
from networks.legacy.monodepth2.layers.skip_decoder import SkipDecoder
from networks.legacy.monodepth2.layers.common import disp_to_depth, get_activation
from networks.legacy.attention_guidance import AttentionGuidance
from networks.legacy.continuous_convolution import ContinuousFusion

from utils.depth import depth2inv

########################################################################################################################


class AdaptiveMultiModalWeighting(nn.Module):

    def __init__(self, in_channels, n_modalities, activation):
        super().__init__()

        self.branchs = nn.ModuleDict()
        for i in range(n_modalities):

            branch = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    activation(inplace=True),
                    nn.AdaptiveAvgPool2d(1), #Global Average Pooling
                    nn.Conv2d(in_channels, 1, kernel_size=1, padding=0, bias=True),
                    activation(inplace=True)
                )

            self.branchs.update({f'branch_{i}': branch})

        self.last_conv = nn.Conv2d(n_modalities, n_modalities, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, modalities_features):

        branch_outs = []
        for i,x in enumerate(modalities_features):
            branch_outs.append(self.branchs[f'branch_{i}'](x))

        x = torch.cat(branch_outs, 1)
        x = self.relu(self.last_conv(x))

        weights = self.softmax(x)

        return  weights



class TeacherGuidedDepthResNet(nn.Module):
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
                 inverse_lidar_input=True, n_power_iterations=5, no_maxpool=False, upsample_path='direct',
                 self_teaching=False, **kwargs):

        super().__init__()

        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        assert guidance in ['attention', 'continuous']

        self.self_teaching = self_teaching
        self.inverse_lidar_input = inverse_lidar_input

        activation_cls = get_activation(activation)

        # keeping the name `encoder` so that we can use pre-trained weight directly
        self.encoder = ResnetEncoder(num_layers=num_layers, input_channels=input_channels, activation=activation_cls,
                                     n_power_iterations=n_power_iterations)
        self.lidar_encoder = ResnetEncoder(num_layers=num_layers, input_channels=1, activation=activation_cls,
                                           no_first_norm=True, no_maxpool=no_maxpool,
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

        if self.self_teaching:
            self.adaptive_weighting = AdaptiveMultiModalWeighting(in_channels=self.num_ch_enc[-1],
                                                                  n_modalities=2,
                                                                  activation=activation_cls)


        self.decoder = DepthDecoder(num_ch_enc=self.num_ch_skips, activation=activation_cls,
                                    uncertainty=self.self_teaching, **kwargs)
        self.cam_decoder = SkipDecoder(num_ch_enc=self.num_ch_skips, activation=activation_cls,
                                         upsample_path=upsample_path, **kwargs)
        self.lidar_decoder = SkipDecoder(num_ch_enc=self.num_ch_skips, activation=activation_cls,
                                         upsample_path=upsample_path, **kwargs)

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

        if self.self_teaching:
            uncertainties = [x[('uncertainty', i)] for i in range(4)]

        cam_disp = self.cam_decoder(cam_features)
        cam_disp = self.scale_inv_depth(cam_disp)[0]
        lidar_disp = self.lidar_decoder(lidar_features)
        lidar_disp = self.scale_inv_depth(lidar_disp)[0]

        if self.training:

            outputs = {
                'inv_depths': [self.scale_inv_depth(d)[0] for d in disps],
                'cam_disp': cam_disp,
                'lidar_disp': lidar_disp,
            }

            if self.self_teaching:
                weights = self.adaptive_weighting([cam_features[-1], lidar_features[-1]])
                outputs['adaptive_weights'] = weights
                outputs['uncertainties'] = uncertainties

        else:

            outputs = {
                'inv_depths': self.scale_inv_depth(disps[0])[0],
                'cam_disp': cam_disp,
                'lidar_disp': lidar_disp,
            }

            if self.self_teaching:
                outputs['uncertainties'] = uncertainties[0]

        return outputs