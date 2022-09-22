from lidartouch.networks.layers.basic_blocks import conv1x1

import torch.nn as nn

class ModalitiesEqualizer(nn.Module):

    def __init__(self, lidar_in_chans, camera_in_chans, activation_cls):
        super().__init__()

        self.activation = activation_cls(inplace=True)

        max_features = max(lidar_in_chans, camera_in_chans)
        min_features = min(lidar_in_chans, camera_in_chans)

        self.equalize_state = 0
        # equalize feature tensor channels to biggest
        if lidar_in_chans != camera_in_chans:
            self.equalizer_conv = conv1x1(min_features, max_features, bias=True)

            if lidar_in_chans > camera_in_chans:
                self.equalize_state = 1
            else:
                self.equalize_state = 2

    def forward(self, cam_input, lidar_input):

        if self.equalize_state == 1:
            cam_input = self.activation(self.equalizer_conv(cam_input))
        elif self.equalize_state == 2:
            lidar_input = self.activation(self.equalizer_conv(lidar_input))
        else:
            pass

        return cam_input, lidar_input


