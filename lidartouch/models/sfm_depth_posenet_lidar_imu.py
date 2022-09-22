from lidartouch.models.sfm_depth_posenet_lidar import SfMDepthPoseNetLiDAR

from lidartouch.utils.identifiers import GT_TRANS_MAG

import hydra

class SfMDepthPoseNetLiDARIMU(SfMDepthPoseNetLiDAR):

    def __init__(self, pose_loss: dict,  *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.pose_loss = hydra.utils.instantiate(pose_loss)


    def compute_losses_and_metrics(self, batch, predictions):
        losses, metrics, reconstruction_loss_outputs = super().compute_losses_and_metrics(batch, predictions)

        # estimated_depths, target_view, source_views, poses, K,
        imu_loss, imu_metrics = self.pose_loss(predictions['poses'], batch[GT_TRANS_MAG])

        gt_pose_magnitudes = {}
        for i, pose_mag in enumerate(batch[GT_TRANS_MAG]):
            gt_pose_magnitudes[f'train/gt_pose_magnitude_{i}'] = pose_mag[0]
            gt_pose_magnitudes[f'train/pose_magnitude_ratio_{i}'] = metrics[f'train/pose_magnitude_{i}'] / pose_mag[0]

        total_loss = losses + imu_loss
        total_metrics = {**metrics, **imu_metrics, **gt_pose_magnitudes}

        return total_loss, total_metrics, reconstruction_loss_outputs




