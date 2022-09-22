import torch

from lidartouch.losses.loss_base import LossBase

class IMUPriorLoss(LossBase):
    """
    IMU prior loss on estimated pose

    Args:
        kwargs: Extra parameters
    """

    def __init__(self, velocity_loss_weight=0.05, **kwargs):
        super().__init__(**kwargs) # forwards all unused arguments

        self.velocity_loss_weight = velocity_loss_weight

    def forward(self, pred_poses, translation_magnitudes):
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

        pred_trans = [pose.mat[:, :3, -1].norm(dim=-1) for pose in pred_poses]
        pred_trans = torch.stack(pred_trans, dim=1)

        """
        n = translation_magnitudes.numel()
        if failure_checks is not None:
            n = failure_checks.sum()
            translation_magnitudes = translation_magnitudes * failure_checks
            pred_trans = pred_trans * failure_checks
        """

        # Calculate velocity supervision loss
        # loss = (pred_trans - translation_magnitudes).abs().sum() / (n + 1e-6)
        # loss = (pred_trans - translation_magnitudes).abs().mean()

        loss = sum([(pred - gt).abs().mean()
                    for pred, gt in zip(pred_trans, translation_magnitudes)]) / len(translation_magnitudes)

        self.add_metric('imu_prior_loss', loss)

        imu_loss = loss * self.velocity_loss_weight

        return imu_loss, self.metrics