from lidartouch.models.base_sfm_depth import BaseSfMDepth

from lidartouch.utils.identifiers import GT_POSES
from lidartouch.utils.pose import Pose

import torch
from lidartouch.utils.depth import inv2depth, compute_depth_metrics
from lidartouch.utils.identifiers import INV_DEPTHS, GT_DEPTH

class SfMDepthGTPose(BaseSfMDepth):

    def __init__(self, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.alpha = 1.0

    def compute_poses(self, batch):
        """Get pose from batch

        It is expected that batch[POSES] is a transformation matrix torch.Tensor of shape [B,S,4,4]
        with B batch size and S the number of source views
        """
        poses = batch[GT_POSES].float()
        poses = [Pose.from_vec(poses[:, i], 'euler') for i in range(poses.shape[1])]

        for pose in poses:
            pose.translation = pose.translation * self.alpha

        return {'poses': poses}

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

        mask = batch[GT_DEPTH][0] > 0
        metrics['median_diff'] = torch.abs(depth[0][mask].median() - batch[GT_DEPTH][0][mask].median())

        result = {
            'metrics': metrics,
            'inv_depth': inv_depth,
            'depth': depth,
        }

        return result
