# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/losses/velocity_loss.py

import torch
import torch.nn as nn

from utils.image import match_scales
from utils.depth import inv2depth

from losses.loss_base import LossBase

class GlobalScalingLoss(LossBase):
    """
    Velocity loss for pose translation.
    """
    def __init__(self, supervised_num_scales=4, **kwargs):
        super().__init__()

        self.n = supervised_num_scales

    def forward(self, inv_depths, gt_depth, **kwargs):
        """
        Calculates velocity loss.
        Parameters
        ----------
        pred_poses : list of Pose
            Predicted pose transformation between origin and reference
        translation_magnitudes : list of Pose
            Ground-truth pose transformation magnitudes between target and source images
        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """

        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        gt_depths = match_scales(gt_depth, depths, self.n)

        losses = []
        for i in range(self.n):
            batch_losses = []
            for gt, p in zip(gt_depths[i], depths[i]):
                gt_masked = gt[gt>0.]
                gt_med = gt_masked.median() + 1e-6
                p_med = p.median() + 1e-6

                loss = (1 - min(p_med/gt_med, gt_med/p_med))
                batch_losses.append(loss)
            losses.append(sum(batch_losses) / len(batch_losses))

        loss = sum(losses) / self.n

        self.add_metric('global_scaling', loss)
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }