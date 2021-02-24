import torch.nn as nn

from functools import partial

from networks.legacy.custom.layers.multi_scale_depth_decoder import MultiScaleDepthDecoder
from networks.legacy.custom.layers.dilated_resnet import resnet18
from networks.legacy.monodepth2.layers.depth_decoder import DepthDecoder
from networks.legacy.custom.layers.dilated_pack_encoder import resnet18 as pack_resnet18
from networks.legacy.custom.layers.depth_pack_decoder import DepthPackDecoder
from networks.legacy.custom.layers.sparse_conv_encoder import SparseConvEncoder, SparseConv1x1
from networks.legacy.custom.layers.skip_decoder import SkipDecoder
from networks.legacy.monodepth2.layers.common import disp_to_depth, get_activation
from networks.legacy.attention_guidance import AttentionGuidance

from utils.depth import depth2inv

########################################################################################################################


class GuidedSparseDepthResNet(nn.Module):
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
    def __init__(self, input_channels=3, activation='relu', guidance='attention', attention_scheme='res-sig',
                 inverse_lidar_input=True, dilation_rates=None, combination='sum', fusion_batch_norm=True,
                 rgb_dilation=True, rgb_no_maxpool=False, lidar_small=False, rgb_strided=False, packing=False,
                 multi_scale=False, decoder='skip', **kwargs):
        super().__init__()

        assert guidance in ['attention', 'continuous']

        self.inverse_lidar_input = inverse_lidar_input
        self.lidar_small = lidar_small

        activation_cls = get_activation(activation)

        self.multi_scale = multi_scale
        self.packing = packing

        if self.packing:
            self.encoder = pack_resnet18(activation_cls, input_channels=input_channels, dilation=rgb_dilation)
        else:
            self.encoder = resnet18(activation_cls, input_channels=input_channels, dilation=rgb_dilation,
                                    no_maxpool=rgb_no_maxpool, strided=rgb_strided, small=self.lidar_small)

        self.lidar_encoder = SparseConvEncoder([2,2,2,2], activation_cls, small=lidar_small,
                                               dilation_rates=dilation_rates, combination=combination)

        self.num_ch_enc = self.encoder.num_ch_enc

        self.extend_lidar = nn.ModuleDict()
        for i in range(len(self.num_ch_enc )):
            in_chans = self.lidar_encoder.num_ch_enc[i]
            out_chans = self.num_ch_enc[i]

            self.extend_lidar.update({
                f"extend_lidar{i}": SparseConv1x1(in_chans, out_chans ,activation_cls)
            })

        skip_features_factor = 2 if ('concat' in attention_scheme) else 1
        self.num_ch_skips = [skip_features_factor * num_ch for num_ch in self.num_ch_enc]

        # at each resblock fuse with guidance the features of both encoders
        self.guidances = nn.ModuleDict()
        for i in range(len(self.num_ch_enc)):

            num_ch =  self.num_ch_enc[i]

            if guidance == 'attention':
                guidance_module = AttentionGuidance(num_ch, activation_cls, attention_scheme, 
                                                    use_batch_norm=fusion_batch_norm)
                self.guidances.update({f"guidance_{i}": guidance_module})
            else:
                print(f"guidance {guidance} not implemented")

        self.decoder_type = decoder

        assert self.decoder_type in ['skip', 'packing', 'iterative']

        if self.packing or self.decoder_type == 'packing':
            self.decoder = DepthPackDecoder(num_ch_enc=self.num_ch_skips, activation=activation_cls, **kwargs)
        elif self.multi_scale and self.decoder_type == 'iterative':
            if self.lidar_small:
                self.decoder = MultiScaleDepthDecoder(num_ch_enc=self.num_ch_skips, activation=activation_cls, **kwargs)
            else:
                self.decoder = DepthDecoder(num_ch_enc=self.num_ch_skips, activation=activation_cls, **kwargs)
        else:
            self.decoder = SkipDecoder(num_ch_enc=self.num_ch_skips, activation=activation_cls, **kwargs)


        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=120.0)

    def forward(self, cam_input, lidar_input):
        """
        Runs the network and returns inverse depth maps
        """

        nb_features = len(self.num_ch_enc)

        if self.inverse_lidar_input:
            lidar_input = depth2inv(lidar_input)

        cam_features = self.encoder(cam_input)
        lidar_features = self.lidar_encoder(lidar_input)

        extended_lidar_features = [self.extend_lidar[f"extend_lidar{i}"](lidar_features[i])
                                   for i in range(nb_features)]
        lidar_features = extended_lidar_features

        self.guided_features = []
        for i in range(nb_features):
            guided_feature = self.guidances[f"guidance_{i}"](cam_features[i], lidar_features[i])
            self.guided_features.append(guided_feature)

        preds = self.decoder(self.guided_features)

        outputs = {}

        if not self.multi_scale:
            keys = ['disp', 'coarse_disp']
            for key in keys:
                if key in preds:
                    outputs[key] = self.scale_inv_depth(preds[key])[0]
            if 'uncertainty' in preds:
                outputs['uncertainty'] = preds['uncertainty']

        else:
            disps = [preds[('disp', i)] for i in range(4)]
            uncertainties = [preds[('uncertainty', i)] for i in range(4) if ('uncertainty', i) in preds]

            if self.training:
                outputs['inv_depths'] = [self.scale_inv_depth(d)[0] for d in disps]
                if len(uncertainties)>0:
                    outputs['uncertainties'] = uncertainties
            else:
                outputs['inv_depths'] = self.scale_inv_depth(disps[0])[0]
                if len(uncertainties) > 0:
                    outputs['uncertainties'] = uncertainties[0]

        return outputs