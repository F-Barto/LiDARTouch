import torch.nn as nn

class NetworkBase(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(self.require_lidar_input, bool)
        assert isinstance(self.require_image_input, bool)

    @property
    def require_lidar_input(self):
        raise NotImplementedError

    @property
    def require_image_input(self):
        raise NotImplementedError

