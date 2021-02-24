# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/losses/velocity_loss.py

from losses.loss_base import LossBase

class VelocityLoss(LossBase):
    """
    Velocity loss for pose translation.
    """
    def __init__(self, velocity_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)

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
            # Calculate velocity supervision loss
            loss = sum([(pred - gt).abs().mean()
                        for pred, gt in zip(pred_trans, translation_magnitudes)]) / len(translation_magnitudes)
            self.add_metric('velocity_loss', loss)
            return {
                'loss': loss.unsqueeze(0) * self.velocity_loss_weight,
                'metrics': self.metrics,
            }