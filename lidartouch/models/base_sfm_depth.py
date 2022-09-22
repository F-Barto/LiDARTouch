from lidartouch.utils.identifiers import SOURCE_VIEWS
from lidartouch.models.base_depth_estimation import BaseDepthEstimationModel

import hydra

class BaseSfMDepth(BaseDepthEstimationModel):
    def __init__(self, reconstruction_loss: dict, *args, **kwargs):
        """

        Args:
            reconstruction_loss:
        """
        super().__init__(*args, **kwargs)

        self.reconstruction_loss = hydra.utils.instantiate(reconstruction_loss)

    def compute_losses_and_metrics(self, batch, predictions):

        smoothness_loss, smoothness_metrics = super().compute_losses_and_metrics(batch, predictions)

        # inputs: estimated_depths, target_view_original, source_views_original, poses, K (, sparse_depth)
        reconstruction_loss_outputs = self.reconstruction_loss(**batch, **predictions)

        photo_loss = reconstruction_loss_outputs.pop('photo_loss')
        photo_metrics = reconstruction_loss_outputs.pop('metrics')

        return photo_loss + smoothness_loss, {**photo_metrics, **smoothness_metrics}, reconstruction_loss_outputs


    def forward(self, batch):

        outputs = {}

        depth_outputs = self.compute_inv_depths(batch)
        outputs.update(depth_outputs)

        if SOURCE_VIEWS in batch:
            pose_outputs = self.compute_poses(batch)
            outputs.update(pose_outputs)

        if not self.training:
            return outputs
        else:
            loss, metrics, other_outputs = self.compute_losses_and_metrics(batch, outputs)
            outputs.update(other_outputs)

        return {**outputs, 'loss': loss, 'metrics': metrics}

    def compute_poses(self, batch):
        raise NotImplementedError
