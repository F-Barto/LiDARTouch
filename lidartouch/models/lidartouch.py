from lidartouch.models.sfm_depth_gtpose import SfMDepthGTPose

import hydra

class LiDARTouch(SfMDepthGTPose):

    def __init__(self, depth_loss, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depth_loss = hydra.utils.instantiate(depth_loss)


    def compute_losses_and_metrics(self, batch, predictions):
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

        total_loss = photo_loss + depth_loss
        metrics = {**photo_metrics, **depth_metrics}

        return total_loss, metrics, reconstruction_loss_outputs
