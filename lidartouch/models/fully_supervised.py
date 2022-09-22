from lidartouch.models.base_depth_estimation import BaseDepthEstimationModel

import hydra

class FullySupervised(BaseDepthEstimationModel):

    def __init__(self, depth_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depth_loss = hydra.utils.instantiate(depth_loss)

    def compute_losses_and_metrics(self, batch, predictions):

        smoothness_loss, smoothness_metrics = super().compute_losses_and_metrics(batch, predictions)

        depth_loss, depth_metrics = self.depth_loss(**batch, **predictions)

        total_loss = smoothness_loss + depth_loss
        metrics = {**smoothness_metrics, **depth_metrics}

        return total_loss, metrics

    def forward(self, batch):

        outputs = {}

        depth_outputs = self.compute_inv_depths(batch)
        outputs.update(depth_outputs)

        if not self.training:
            return outputs
        else:
            loss, metrics = self.compute_losses_and_metrics(batch, outputs)

        return {**outputs, 'loss': loss, 'metrics': metrics}
