from lidartouch.losses.reconstruction.ssim_l1 import SSIM_L1
from lidartouch.utils.image import match_scales, match_scale
from lidartouch.utils.depth import inv2depth

import einops
import torch
from collections import defaultdict

class Hinted_SSIM_L1(SSIM_L1):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_outputs(self, estimated_depth, target_view, source_views, poses, K, sparse_depth):

        outputs_by_source = super().compute_outputs(estimated_depth, target_view, source_views, poses, K)
        outputs_by_source['hinted'] = self.warp_and_compare(sparse_depth, target_view, source_views, poses, K)

        # we add mask indicating presence of depth data so that pixels missing depth data are never chosen as minimum
        has_depth_data_mask = sparse_depth > 0.  # B,1,H,W
        has_depth_data_mask = einops.repeat(has_depth_data_mask, 'b c h w -> l b c h w', l=len(source_views))

        if self.reprojection_masking:
            valid_masks = outputs_by_source['hinted']['valid_masks'] & has_depth_data_mask
        else:
            valid_masks = has_depth_data_mask

        outputs_by_source['hinted']['valid_masks'] = valid_masks

        return outputs_by_source


    def forward(self, inv_depths, target_view_original, source_views_original, poses, intrinsics, sparse_depth_original,
                **kwargs):
        """
        Calculates training supervised loss.

        Args:
            estimated_depths : Predicted depth maps for the source image at each prediction scale;
                list of tensor [B,1,H_i,W_i]

        Returns:
            outputs: Output dictionary
        """

        estimated_depths = [inv2depth(inv_depths[i]) for i in range(self.num_scales)]

        sparse_depths = match_scales(sparse_depth_original, estimated_depths, self.num_scales)
        target_view = match_scales(target_view_original, estimated_depths, self.num_scales)

        outputs_by_scales = defaultdict(list)

        for i in range(self.num_scales):

            outputs_by_source = self.compute_outputs(estimated_depths[i], target_view[i], source_views_original,
                                                     poses, intrinsics, sparse_depths[i])

            # outputs_by_source is updated with reduced losses
            outputs_by_source = self.reduce_across_views(outputs_by_source)

            masks = self.compute_masks(outputs_by_source, sparse_depth=sparse_depths[i])
            mask = torch.stack(list(masks.values()), 0).all(0, keepdim=False)  # aggregates masks with AND operation
            outputs_by_scales['mask'].append(mask)

            masked_outputs = self.apply_mask(outputs_by_source, mask)

            # aggregate computations for each scale
            for k, v in masked_outputs.items():
                outputs_by_scales[k].append(v)

            for k, v in masks.items():
                outputs_by_scales[k].append(v)

            for source, outputs in outputs_by_source.items():
                if not (source in outputs_by_scales):
                    outputs_by_scales[source] = defaultdict(list)
                for k, v in outputs.items():
                    outputs_by_scales[source][k].append(v)

        # final reduce and return
        outputs = self.reduce_across_scales(outputs_by_scales)
        outputs['outputs_by_scales'] = outputs_by_scales
        outputs['metrics'] = self.metrics

        return outputs




