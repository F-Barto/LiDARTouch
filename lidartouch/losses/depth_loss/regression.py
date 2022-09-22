import torch
import torch.nn as nn

import sys

from lidartouch.utils.image import match_scales
from lidartouch.losses.loss_base import LossBase
from lidartouch.utils.depth import depth2inv
########################################################################################################################

class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        """
        Initializes the BerHuLoss class.
        Parameters
        ----------
        threshold : float
            Mask parameter
        """
        super().__init__()
        self.threshold = threshold
    def forward(self, pred, gt):
        """
        Calculates the BerHu loss.
        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted inverse depth map
        gt : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth map
        Returns
        -------
        loss : torch.Tensor [1]
            BerHu loss
        """
        huber_c = torch.max(pred - gt)
        huber_c = self.threshold * huber_c
        diff = (pred - gt).abs()

        # Remove
        # mask = (gt > 0).detach()
        # diff = gt - pred
        # diff = diff[mask]
        # diff = diff.abs()

        huber_mask = (diff > huber_c).detach()
        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2
        return torch.cat((diff, diff2)).mean()

class SilogLoss(nn.Module):
    def __init__(self, ratio=10, ratio2=0.85):
        super().__init__()
        self.ratio = ratio
        self.ratio2 = ratio2

    def forward(self, pred, gt):
        log_diff = torch.log(pred * self.ratio) - \
                   torch.log(gt * self.ratio)
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.ratio2 * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * self.ratio
        return silog_loss

########################################################################################################################

def get_loss_func(regression_method, **kwargs):
    """Determines the regression loss to be used, given the regression method."""
    if regression_method.endswith('l1'):
        return nn.L1Loss()
    elif regression_method.endswith('mse'):
        return nn.MSELoss()
    elif regression_method.endswith('berhu'):
        return BerHuLoss(**kwargs)
    elif regression_method.endswith('silog'):
        return SilogLoss(**kwargs)
    elif regression_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    else:
        raise ValueError('Unknown regression loss {}'.format(regression_method))

########################################################################################################################

class RegressionLoss(LossBase):
    """
    regression loss for inverse depth maps.
    Parameters
    ----------
    regression_method : str
        Which regression method will be used
    num_scales : int
        Number of scales used by the regression loss
    kwargs : dict
        Extra parameters
    """
    def __init__(self, regression_method='sparse-l1', num_scales=4, depth_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_func = get_loss_func(regression_method, **kwargs)
        self.regression_method = regression_method
        self.num_scales = num_scales
        self.depth_loss_weight = depth_loss_weight


    def calculate_losses(self, inv_depths, gt_depths, valid_masks=None):
        """
        Calculate the regression loss.
        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth maps
        Returns
        -------
        loss : torch.Tensor [1]
            Average regression loss for all scales
        """

        losses = []

        gt_inv_depths = [depth2inv(gt_depths[i]) for i in range(self.num_scales)]


        for i in range(self.num_scales):
            mask = None

            if self.regression_method.startswith('sparse'):
                mask = (gt_depths[i] > 0.).detach()

            if valid_masks is not None:
                mask = mask & valid_masks[i] if mask is not None else valid_masks[i]

            inv_depth = inv_depths[i]
            gt_inv_depth = gt_inv_depths[i]

            if mask is not None:
                inv_depth = inv_depth[mask]
                gt_inv_depth = gt_inv_depth[mask]

            loss = self.loss_func(inv_depth, gt_inv_depth)

            if torch.isnan(loss):
                # if mask is all false
                # -> masked_inv_depth & masked_gt_inv_depth == Tensor([]) -> loss_func returns nan
                # Such case is okay, otherwise there is an issue with the preds so we print it
                if not len(inv_depth) == 0:
                    print('masked_inv_depth: ', inv_depth, file=sys.stderr)
                    print('len(masked_inv_depth): ', len(inv_depth), file=sys.stderr)
                    print('regression loss: ', loss, file=sys.stderr)
                loss = torch.tensor(0.)

            losses.append(loss)

        # Return per-scale average loss
        return losses

    def forward(self, inv_depths, gt_depth, **kwargs):
        """
        Calculates training regression loss.
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

        # Calculate and store regression loss
        losses = self.calculate_losses(inv_depths, gt_depths)
        loss = sum([losses[i] for i in range(self.num_scales)]) / self.num_scales

        self.add_metric('depth_loss', loss)
        # Return losses and metrics
        return loss * self.depth_loss_weight, self.metrics