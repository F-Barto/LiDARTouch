from utils.depth import calc_smoothness

from losses.loss_base import LossBase

class SmoothnessLoss(LossBase):
    def __init__(self, smooth_loss_weight, scales=4):
        super().__init__()

        self.n = scales
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
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.n)]) / self.n


        # Store and return smoothness loss
        self.add_metric('smoothness_loss', smoothness_loss)
        return smoothness_loss

    def forward(self, inv_depths, images):

        smoothness_loss = self.calc_smoothness_loss(inv_depths, images)

        # Apply smoothness loss weight
        weighted_smoothness_loss = self.smooth_loss_weight * smoothness_loss

        return weighted_smoothness_loss