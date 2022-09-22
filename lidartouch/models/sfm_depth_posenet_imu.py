from lidartouch.models.sfm_depth_posenet import SfMDepthPoseNet
from lidartouch.utils.depth import inv2depth, compute_depth_metrics
from lidartouch.utils.identifiers import INV_DEPTHS, GT_DEPTH, GT_TRANS_MAG

import torch
import hydra

class SfMDepthPoseNetIMU(SfMDepthPoseNet):

    def __init__(self, pose_loss: dict, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.pose_loss = hydra.utils.instantiate(pose_loss)
        self.alpha = 1.0

    def compute_losses_and_metrics(self, batch, predictions):
        photo_loss, photo_metrics, reconstruction_loss_outputs = super().compute_losses_and_metrics(batch, predictions)

        # estimated_depths, target_view, source_views, poses, K,
        imu_loss, imu_metrics = self.pose_loss(predictions['poses'], batch[GT_TRANS_MAG] * self.alpha)

        total_loss = photo_loss + imu_loss
        metrics = {**photo_metrics, **imu_metrics}

        return total_loss, metrics, reconstruction_loss_outputs

    def evaluate_depth(self, batch):
        """
        Evaluate batch to produce depth metrics.

        Returns
        -------
        output : dict
            Dictionary containing a "metrics" and a "inv_depth" key

            metrics : torch.Tensor [7]
                Depth metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)

            inv_depth:
                predicted inverse depth
        """
        # Get predicted depth
        output = self(batch)

        inv_depth = output[INV_DEPTHS][0]
        depth = inv2depth(inv_depth) / self.alpha

        # Calculate predicted metrics
        metrics = compute_depth_metrics(gt=batch[GT_DEPTH], pred=depth, **self.metrics_hparams)
        # Return metrics and extra information

        metrics['median_diff'] = torch.abs(depth[0].median() - batch[GT_DEPTH][0].median())

        result = {
            'metrics': metrics,
            'inv_depth': inv_depth,
            'depth': depth,
        }

        return result
