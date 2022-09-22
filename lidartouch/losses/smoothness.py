from lidartouch.utils.depth import calc_smoothness
from lidartouch.utils.image import match_scales

from lidartouch.losses.loss_base import LossBase

class SmoothnessLoss(LossBase):
    def __init__(self, smooth_loss_weight=0.001, num_scales=4):
        super().__init__()

        self.num_scales = num_scales
        self.smooth_loss_weight = smooth_loss_weight


    def calc_smoothness_loss(self, inv_depths, images):
        """
        Calculates the smoothness loss for inverse depth maps.
        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales
        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.num_scales)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.num_scales)]) / self.num_scales

        return smoothness_loss

    def forward(self, inv_depths, image):
        images = match_scales(image, inv_depths, self.num_scales)

        smoothness_loss = self.calc_smoothness_loss(inv_depths, images)

        # Store smoothness loss
        self.add_metric('smoothness_loss', smoothness_loss)

        # Apply smoothness loss weight
        weighted_smoothness_loss = self.smooth_loss_weight * smoothness_loss

        return weighted_smoothness_loss, self.metrics