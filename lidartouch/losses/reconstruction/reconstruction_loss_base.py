import torch

from lidartouch.utils.camera import Camera
from lidartouch.utils.multiview_warping_and_projection import view_synthesis
from lidartouch.utils.image import match_scales, match_scale
from lidartouch.utils.depth import inv2depth

from lidartouch.losses.loss_base import LossBase

import math
from collections import defaultdict


def str_shapes(l):
    if l is None:
        return 'None'
    s = ''
    for e in l:
        s += f'{e.shape} '
    return s


def masked_min(tensor, mask=None, dim=0, fill_value=math.inf):
    if mask is None:
        return tensor.min(dim, keepdim=False).values

    masked = torch.mul(tensor, mask)
    inf_mask = torch.zeros_like(tensor)
    inf_mask[~mask] = math.inf # highest value possible
    min_tensor = (masked + inf_mask).min(dim, keepdim=False).values
    any_mask = mask.any(dim=dim, keepdim=False) # pixel is False if invalid across all views
    min_tensor[~any_mask] = fill_value

    return min_tensor


class ReconstructionLossBase(LossBase):
    """
    Self-Supervised loss for depth maps estimation through view reconstruction.

    Args:
        kwargs: Extra parameters
    """

    def __init__(self, num_scales=1, reprojection_masking=True, automasking=True, padding_mode='zeros',
                 align_corners=True, **kwargs):
        super().__init__(**kwargs) # forwards all unused arguments

        self.num_scales = num_scales
        self.reprojection_masking = reprojection_masking
        self.automasking = automasking
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def photo_loss(self, source_image, target_image):
        return NotImplementedError

    def warp_ref_image(self, depth, source_image, K, source_K, pose):
        """
        Warps an image from a source view to produce a reconstruction of a target image.

        Args:
            depth: Depth map in the target image reference frame of shape [B,1,DH,DW]
            source_image: RGB image in the source image reference frame of shape [B,3,H,W]
            K: Intrinsics of the camera used to obtain target view image, tensor of shape [B,3,3]
            source_K: Intrinsics of the camera used to obtain source view image.
                If the same camera is used, source_k should be equals to K.
            pose: Camera transformation from target view to source view.

        Returns:
            warped_source: Warped source image (reconstructing the target image), [B,1,DH,DW]
            valid_mask: Binary mask where True indicates a pixel with valid reprojection and False otherwise
        """
        B, _, H, W = source_image.shape
        device = source_image.device

        # Generate cameras
        _, _, DH, DW = depth.shape
        scale_factor = DW / float(W)

        cam = Camera(K=K.float()).scaled(scale_factor).to(device)
        source_cam = Camera(K=source_K.float(), Tcw=pose).scaled(scale_factor).to(device)

        warped_source, valid_mask = view_synthesis(source_image, depth, source_cam, cam,
                                                   padding_mode=self.padding_mode, align_corners=self.align_corners,
                                                   return_valid_mask=True)

        # Return warped reference image
        return warped_source, valid_mask

    def warp_and_compare(self, depth, target_view, source_views, poses, K, **kwargs):
        outputs = {
            'reconstructions': [],       # output shape: [L,B,3,H,W] if input is RGB else [L,B,1,H,W]
            'reconstruction_losses': [], # output shape: [L,B,1,H,W]
            'valid_masks': []            # output shape: [L,B,1,H,W]
        }

        for (source_view, pose) in zip(source_views, poses):
            warped_view, valid_mask = self.warp_ref_image(depth, source_view, K, K, pose)

            # Calculate and store image loss
            photometric_loss = self.photo_loss(warped_view, target_view)

            outputs['reconstructions'].append(warped_view)
            outputs['valid_masks'].append(valid_mask)
            outputs['reconstruction_losses'].append(photometric_loss)

        if not self.reprojection_masking:
            outputs.pop('valid_masks')

        for k,v in outputs.items():
            outputs[k] = torch.stack(v, 0)

        return outputs

    def auto_compare(self, estimated_depth, target_view, source_views, **kwargs):
        outputs = {
            'reconstruction_losses': [],  # output shape: [L,B,1,H,W]
        }

        for source_view in source_views:
            source_view = match_scale(source_view, estimated_depth)
            photometric_loss = self.photo_loss(source_view, target_view)
            outputs['reconstruction_losses'].append(photometric_loss)

        for k,v in outputs.items():
            outputs[k] = torch.stack(v, 0)

        return outputs

    def compute_outputs(self, estimated_depth, target_view, source_views, poses, K, **kwargs):
        """
        Compute outputs by source. Each outputs_by_source[<key>] contains a dict with value for `reconstruction_losses`
        (source view reprojected in target view) and possibly a `value for `valid_masks`, a binary mask indicated which
        pixels have a valid reprojection.

        Each leaf element of the outputs_by_source dict is of shape [L,B,1,H,W] where L is nb of source_views.

        The illustration below depicts the outputs of shape [L,B,1,H,W] from `warp_and_compare` and `auto_compare`
        as well as the organisation of the returned dict `outputs_by_source`.

        Not every source necessary have a valid_masks.

                                            estimated              auto
                                         view1   view2    |    view1    view2  <- L = 2
                                         ____     ____    |    ____     ____
         reconstruction_losses          |    |   |    |   |   |    |   |    |
                                        |____|   |____|   |   |____|   |____|
                                                                          └> shape of [B,1,H,W]
                                         ____     ____    |
         valid_masks                    |    |   |    |   |
                                        |____|   |____|   |

        Where estimated and auto are reprojection sources (from estimated depth and identity respectively)

        """

        outputs_by_source = {}

        outputs_by_source['estimated'] = self.warp_and_compare(estimated_depth, target_view,
                                                               source_views, poses, K)

        if self.automasking:
            outputs_by_source['auto'] = self.auto_compare(estimated_depth, target_view, source_views)

        return outputs_by_source


    def reduce_across_views(self, outputs_by_source):
        """
        elements of outputs_by_source: [L,B,1,H,W] where L is nb of source_views
        elements of valid_masks: [L,B,1,H,W] where L is nb of source_views
        
        Reduce reconstruction losses of each source ('estimated', 'auto', etc...) across views while considering the
        valid masks. Elements for which the mask has value of False are excluded from being selected by the `min` 
        operation.

        The illustration below depicts, roughly, the stream of computation as well as the organisation of the dict
        `outputs_by_source`.
        
                                            estimated                   auto
                                         view1   view2      |       view1    view2  <- L = 2
                                         ____     ____      |        ____     ____
        reconstruction_losses      min( |    |   |    | )   |  min( |    |   |    | )
                                        |____| , |____|     |       |____| , |____|
                                                                              └> shape of [B,1,H,W]
                                         ____     ____      |
        valid_masks                     |    |   |    |     |
                                        |____|   |____|     |
        
                                                ↓                          ↓
                                               ____                       ____
        minimized_loss                        |    |                     |    |
            [L,B,1,H,W] -> [B,1,H,W]          |____|                     |____|

        """

        for source, outputs in outputs_by_source.items():
            reconstruction_losses = outputs.get('reconstruction_losses', None)
            valid_masks = outputs.get('valid_masks', None)

            if reconstruction_losses is None:
                continue

            minimized_loss = masked_min(reconstruction_losses, dim=0, mask=valid_masks)
            outputs['minimized_loss'] = minimized_loss

        return outputs_by_source


    def compute_masks(self, outputs_by_source, **kwargs):

        """
        Any mask should be True if pixel is kept in loss computation (valid pixel) and False otherwise

        elements of minimized_loss: [B,1,H,W]
        elements of valid_masks: [L,B,1,H,W]

        output: 1

        """

        # each element should be of shape [1,B,1,H,W]
        masks = {}

        if self.reprojection_masking:
            # reprojection_mask is True for pixels with a valid reprojection (i.e., falls in the target camera image plane)
            reprojection_mask = outputs_by_source['estimated']['valid_masks']
            # if pixel is valid in any view, set it as valid (True)
            reprojection_mask = reprojection_mask.any(dim=0, keepdim=False)
            masks['reprojection_mask'] = reprojection_mask

        if self.automasking:
            min_auto_photo_loss = outputs_by_source['auto'].get('minimized_loss', None)
            if min_auto_photo_loss is not None:
                # auto_mask: True for pixel if estimated reprojection has lower photometric error than identity reprojection
                auto_mask = outputs_by_source['estimated']['minimized_loss'] < min_auto_photo_loss
                masks['auto_mask'] = auto_mask

        return masks

    def apply_mask(self, outputs_by_source, mask):
        """
        Args:
            outputs_by_source: dict containing all pre-computed losses by source ('estimated', 'auto', ...)
            mask: mask of shape [B,1,H,W] indicating which pixel should be used in final loss computation

        Return:
            output masked and averaged across all dims
        """

        masked_outputs = {}

        photo_loss = outputs_by_source['estimated']['minimized_loss'][mask].mean()
        masked_outputs['photo_loss'] = photo_loss.nan_to_num(nan=0.0) # photo_loss is nan is mask.all() == False

        return masked_outputs


    def reduce_across_scales(self, outputs_by_scales):

        photo_loss = sum(outputs_by_scales['photo_loss']) / self.num_scales

        outputs = {
            'photo_loss': photo_loss
        }

        self.add_metric('photo_loss', photo_loss)

        return outputs


    def forward(self, inv_depths, target_view_original, source_views_original, poses, intrinsics, **kwargs):
        """
        Calculates training supervised loss.

        Args:
            estimated_depths : Predicted depth maps for the source image at each prediction scale;
                list of tensor [B,1,H_i,W_i]

        Returns:
            outputs: Output dictionary
        """

        estimated_depths = [inv2depth(inv_depths[i]) for i in range(self.num_scales)]

        target_view = match_scales(target_view_original, estimated_depths, self.num_scales)

        outputs_by_scales = defaultdict(list)

        for i in range(self.num_scales):

            outputs_by_source = self.compute_outputs(estimated_depths[i], target_view[i], source_views_original,
                                                     poses, intrinsics)
            # outputs_by_source is updated with reduced losses
            outputs_by_source = self.reduce_across_views(outputs_by_source)

            masks = self.compute_masks(outputs_by_source, **kwargs)
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


















