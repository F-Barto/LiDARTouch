from torch import nn
import torch


class AttentionBlock(nn.Module):
    def __init__(self, inplanes, activation_cls, attention_scheme='res-sig', use_batch_norm=True):
        super().__init__()

        self.activation = activation_cls(inplace=True)

        if 'sig' in attention_scheme:
            self.act = nn.Sigmoid()
        elif 'tan' in attention_scheme:
            self.act = nn.Tanh()
        elif 'softmax' in attention_scheme:
            # if we use softmax, it will be used later, so for now just make it the usual conv + bn + activation
            self.act = activation_cls(inplace=True)
        else:
            raise ValueError(f'Last activation choice invalid either sig or tanh: {attention_scheme}')

        if 'concat' in attention_scheme:
            planes = inplanes
        else:
            planes= inplanes//2


        self.conv_1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv_3x3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)

        self.use_batch_norm = use_batch_norm

        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)


        nn.init.kaiming_normal_(self.conv_1x1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.conv_3x3.weight, gain=1.0)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_1x1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)

        x = self.conv_3x3(x)
        if self.use_batch_norm:
            x = self.bn2(x)

        out = self.act(x)

        return out

class AttentionGuidance(nn.Module):
    def __init__(self, inplanes, activation_cls, attention_scheme='res-sig', use_batch_norm=True):
        super().__init__()

        self.attention_scheme = attention_scheme

        if 'concat' == self.attention_scheme:
            pass
        elif 'concat' in self.attention_scheme:
            if 'concatlin' in self.attention_scheme:
                self.pre_conv_3x3 = nn.Conv2d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, bias=True)
                nn.init.kaiming_normal_(self.pre_conv_3x3.weight, mode='fan_out', nonlinearity='relu')

            self.attention_block = AttentionBlock(inplanes * 2, activation_cls, self.attention_scheme)
        else:
            self.lidar_attention_block = AttentionBlock(inplanes * 2, activation_cls, self.attention_scheme,
                                                        use_batch_norm=use_batch_norm)
            self.image_attention_block = AttentionBlock(inplanes * 2, activation_cls, self.attention_scheme,
                                                        use_batch_norm=use_batch_norm)

        if 'preconv' in self.attention_scheme:
            self.preconv_lidar = nn.Sequential(
                nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                activation_cls(inplace=True)
            )

            self.preconv_rgb = nn.Sequential(
                nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                activation_cls(inplace=True)
            )

        if 'softmax' in self.attention_scheme:
            self.softmax = torch.nn.Softmax(dim=0)


    def fuse_features(self, original_features, attentive_masks, lidar_mask=None):
        if 'res' in self.attention_scheme:
            residual_features = [of * am + of for of,am in zip(original_features, attentive_masks)]
            return sum(residual_features)
        elif 'softmax' in self.attention_scheme:

            if 'gsoftmax' in self.attention_scheme:
                # reduce to Bx1xHxW through global average pooling
                attentive_masks = [am.mean([1], keepdim=True) for am in attentive_masks]

            concat_attentive_masks = torch.stack(attentive_masks, dim=0)
            weights = self.softmax(concat_attentive_masks)

            if lidar_mask is not None:
                image_weights = weights[0] * lidar_mask + (1. - lidar_mask)
                lidar_weights = weights[1] * lidar_mask
                weights = torch.stack([image_weights, lidar_weights], dim=0)

            features = [of * w for of, w in zip(original_features, weights)]
            return sum(features)
        elif 'mult' in self.attention_scheme:
            features = [of * am for of, am in zip(original_features, attentive_masks)]
            return sum(features)
        else:
            raise ValueError(f'Attention scheme invalid either res or mult: {self.attention_scheme}')

    def forward(self, image_features, lidar_features, **kwargs):

        lidar_mask=None
        if isinstance(lidar_features, tuple):
            lidar_features, lidar_mask = lidar_features
            if not 'masked' in self.attention_scheme:
                lidar_mask = None

        if 'preconv' in self.attention_scheme:
            lidar_features = self.preconv_lidar(lidar_features)
            image_features = self.preconv_rgb(image_features)

        c = torch.cat([image_features, lidar_features], dim=1)

        if 'concat' == self.attention_scheme:
            final_features = c
        elif 'concatlin' in self.attention_scheme:
            x = self.pre_conv_3x3(c)
            attentive_mask = self.attention_block(x)
            final_features = self.fuse_features([x], [attentive_mask])
        elif 'concat' in self.attention_scheme:
            attentive_mask = self.attention_block(c)
            final_features = self.fuse_features([c], [attentive_mask])
        else:
            image_attentive_mask = self.image_attention_block(c)
            lidar_attentive_mask = self.lidar_attention_block(c)
            attentive_masks = [image_attentive_mask, lidar_attentive_mask]
            final_features = self.fuse_features([image_features, lidar_features], attentive_masks, lidar_mask)

        return final_features

