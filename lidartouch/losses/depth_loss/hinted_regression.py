import torch
import torch.nn as nn

import sys

from lidartouch.utils.image import match_scales
from lidartouch.losses.depth_loss.regression import RegressionLoss
from lidartouch.utils.depth import depth2inv

class HintedRegressionLoss(RegressionLoss):
    """
    Supervised loss for inverse depth maps.
    Parameters
    ----------
    supervised_method : str
        Which supervised method will be used
    num_scales : int
        Number of scales used by the supervised loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



    def forward(self, inv_depths, gt_depth, estimated_minimized_loss, hinted_minimized_loss, reprojection_mask=None,
                **kwargs):
        """
        Calculates training supervised loss.
        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        gt_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the original image
        return_logs : bool
            True if logs are saved for visualization
        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """

        # Match predicted scales for ground-truth
        gt_depths = match_scales(gt_depth, inv_depths, self.num_scales, mode='nearest')

        valid_masks = []
        for i in range(self.num_scales):
            mask = hinted_minimized_loss[i] < estimated_minimized_loss[i]
            if reprojection_mask is not None:
                mask = mask & reprojection_mask[i]
            valid_masks.append(mask.detach())



        # Calculate and store supervised loss
        losses = self.calculate_losses(inv_depths, gt_depths,valid_masks=valid_masks)
        loss = sum([losses[i] for i in range(self.num_scales)]) / self.num_scales

        self.add_metric('depth_loss', loss)
        # Return losses and metrics
        return loss * self.depth_loss_weight, self.metrics