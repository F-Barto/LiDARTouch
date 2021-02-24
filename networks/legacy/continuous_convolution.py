import torch
import torch.nn as nn
from networks.legacy.pyramidal_feature_combination import PyramidalFeatureCombination

class ParametricContinuousConv(nn.Module):
    """Layer to perform a convolution followed by activation function
    """
    def __init__(self, channels, activation):
        super(ParametricContinuousConv, self).__init__()

        self.mlp = nn.Sequential(
            torch.nn.Linear(3, channels // 2),
            activation(inplace=True),
            torch.nn.Linear(channels // 2, channels),
            activation(inplace=True)
        )

        self.conv = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)

        self.activation = activation(inplace=True)

    def forward(self, feature_tensor, nn_diff_pts_3d, pixel_idxs, nn_pixel_idxs):

        b, n, k, _ = nn_diff_pts_3d.shape

        ##################### Extracting feature vectors from feature tensors #####################
        flat_nn_pixel_idxs = nn_pixel_idxs.reshape(b, n * k, 2).long().detach() # pytorch requires long for idxs
        f_flat = feature_tensor.permute(0, 2, 3, 1)[:, flat_nn_pixel_idxs[:, :, 1], flat_nn_pixel_idxs[:, :, 0]]
        f = f_flat.reshape(b, n, k, -1) # of shape B x N x K x C

        # nn_diff_pts_3d is of shape B x N x K x 3 out is of shape B x N x K x C
        out = self.mlp(nn_diff_pts_3d)

        ##################### Parametric continuous conv through MLP #####################

        # MLP(x_i - x_k) * f_k
        out = out * f

        # sum across neighbor dim -> B x N x 1 x C
        out = out.sum(dim=2, keepdim=True)

        # permute B x N x 1 x C   to   B x C x N x 1 before conv
        out = self.conv(out.permute(0, 3, 1, 2))
        out = self.activation(self.bn(out))

        # squeeze and permute B x C x N x 1  to   B x C X 1 x N
        out = out.permute(0, 1, 3, 2)

        ##################### Populating output feature map #####################
        h = torch.zeros(feature_tensor.shape, dtype=feature_tensor.dtype).to(device=feature_tensor.device)
        pixel_idxs = pixel_idxs.long() # pytorch requires long for idxs
        h[:, :, pixel_idxs[:, :, 1], pixel_idxs[:, :, 0]] = out

        return h

class SequentialPametricContinuousConv(nn.Module):
    def __init__(self, in_channels, activation_cls, nb_blocks):
        super().__init__()

        self.nb_blocks = nb_blocks

        self.cont_convs = nn.ModuleDict()
        for i in range(nb_blocks):
            self.cont_convs.update({f"cont_conv_{i}": ParametricContinuousConv(in_channels, activation_cls)})

    def forward(self, features, nn_diff_pts_3d, pixel_idxs, nn_pixel_idxs):

        for i in range(self.nb_blocks):
            features = self.cont_convs[f"cont_conv_{i}"](features, nn_diff_pts_3d, pixel_idxs, nn_pixel_idxs)

        return features

class ContinuousFusion(nn.Module):
    def __init__(self, in_channels, activation_cls, dilation_rates=[1,2,8,16], combination='sum'):
        super().__init__()

        self.multi_scale_conv = PyramidalFeatureCombination(in_channels, in_channels, activation_cls,
                                                            dilation_rates=dilation_rates, combination=combination)

        self.continuous_conv = SequentialPametricContinuousConv(in_channels, activation_cls, nb_blocks=2)

        self.conv = self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, image_features, lidar_features, nn_diff_pts_3d, pixel_idxs, nn_pixel_idxs):

        c = torch.cat([image_features, lidar_features], dim=1)

        multi_scale_out = self.multi_scale_conv(c)

        cont_out = self.continuous_conv(c, nn_diff_pts_3d, pixel_idxs, nn_pixel_idxs)

        out = c + cont_out + multi_scale_out

        final_features = self.conv(out)

        return final_features
