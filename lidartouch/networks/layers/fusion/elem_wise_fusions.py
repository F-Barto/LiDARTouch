from lidartouch.networks.layers.fusion.channels_equalizer import ModalitiesEqualizer
from lidartouch.networks.layers.fusion.fusion_base import FusionBase

class ElemWiseMultFusion(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, activation_cls=None, **kwargs):
        super().__init__(**kwargs)

        self.activation = None
        self.equalizer = None
        if self.lidar_in_chans is not None and self.image_in_chans is not None and activation_cls is not None:
            self.setup_module(lidar_in_chans, image_in_chans, activation_cls)

    def setup_module(self, lidar_in_chans, image_in_chans, activation_cls):
        self.activation = activation_cls(inplace=True)
        self.equalizer = ModalitiesEqualizer(lidar_in_chans, image_in_chans, activation_cls)

    @property
    def require_chans(self):
        return True

    @property
    def require_activation(self):
        return True

    def forward(self, image_features, lidar_features):
        image_features, lidar_features = self.equalizer(image_features, lidar_features)
        return image_features * lidar_features


class ElemWiseSumFusion(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, activation_cls=None, **kwargs):
        super().__init__(**kwargs)

        self.activation = None
        self.equalizer = None
        if lidar_in_chans is not None and image_in_chans is not None and activation_cls is not None:
            self.setup_module(lidar_in_chans, image_in_chans, activation_cls)

    def setup_module(self, lidar_in_chans, image_in_chans, activation_cls):
        self.activation = activation_cls(inplace=True)
        self.equalizer = ModalitiesEqualizer(lidar_in_chans, image_in_chans, activation_cls)

    @property
    def require_chans(self):
        return True

    @property
    def require_activation(self):
        return True

    def forward(self, image_features, lidar_features):
        image_features, lidar_features = self.equalizer(image_features, lidar_features)
        return image_features + lidar_features