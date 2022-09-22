from lidartouch.models.sfm_depth_posenet import SfMDepthPoseNet

from lidartouch.utils.image import match_scales
from lidartouch.utils.depth import depth2inv

import hydra


def compute_median(gt_depth, inv_depths):

    n = len(inv_depths)
    gt_depths = match_scales(gt_depth, inv_depths, n, mode='nearest')
    gt_inv_depths = [depth2inv(gt_depths[i]) for i in range(n)]

    metrics = {}

    for i in range(n):

        mask = (gt_depths[i] > 0.).detach()

        inv_depth = inv_depths[i]
        gt_inv_depth = gt_inv_depths[i]

        metrics[f'train/median_masked_invdepth_{i}'] = inv_depth[mask][0].median()
        metrics[f'train/median_gt_invdepth_{i}'] = gt_inv_depth[mask][0].median()
        metrics[f'train/median_masked_invdepth_ration_{i}'] = (inv_depth / gt_inv_depth)[mask][0].median()
        metrics[f'train/median_invdepth_{i}'] = inv_depth[0].median()


    return metrics


class SfMDepthPoseNetLiDAR(SfMDepthPoseNet):

    def __init__(self, depth_loss: dict, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.depth_loss = hydra.utils.instantiate(depth_loss)

        self.depth_w = 1.0


    def compute_losses_and_metrics(self, batch, predictions):

        #batch['sparse_depth_original'] = batch['sparse_depth_original'] * self.depth_w
        #batch['sparse_depth'] = batch['sparse_depth'] * self.depth_w

        photo_loss, photo_metrics, reconstruction_loss_outputs = super().compute_losses_and_metrics(batch, predictions)

        # flattening output dict:
        flat_reconstruction_loss_outputs = {}
        outputs_by_scales = reconstruction_loss_outputs['outputs_by_scales']
        for source, output in outputs_by_scales.items():
            if isinstance(output, dict):
                for element, val in output.items():
                    flat_reconstruction_loss_outputs[f'{source}_{element}'] = val
            else:
                flat_reconstruction_loss_outputs[source] = output

        # estimated_depths, target_view, source_views, poses, K,
        depth_loss, depth_metrics = self.depth_loss(**batch, **predictions, **flat_reconstruction_loss_outputs)

        depth_medians = compute_median(batch['gt_depth'], predictions['inv_depths'])

        pose_magnitudes = {}
        for i, pose in enumerate(predictions['poses']):
            pose_magnitudes[f'train/pose_magnitude_{i}'] = pose.translation[0].norm()


        total_loss = photo_loss + depth_loss * self.depth_w # self.depth_w * depth_loss
        metrics = {**photo_metrics, **depth_metrics, 'depth_w': self.depth_w, **depth_medians, **pose_magnitudes}

        return total_loss, metrics, reconstruction_loss_outputs


