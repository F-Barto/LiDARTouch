# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch

from utils.camera import Camera
from utils.multiview_warping_and_projection import view_synthesis
from utils.image import match_scales
from utils.depth import inv2depth

from losses.loss_base import LossBase
from losses.handlers.handler_base import LossHandler

import sys

class MultiViewLossHandler(LossHandler, LossBase):
    """
    Semi-Supervised loss for inverse depth maps.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """

    def __init__(self, losses_hparams):
        super().__init__(losses_hparams=losses_hparams)

        losses = self.parse_losses(['photo', 'smoothness', 'hinted'])

        print('losses_hparams')
        print(self.losses_hparams)

        if hasattr(losses_hparams, 'masked'):
            self.masked = losses_hparams.masked
        else:
            self.masked = False



        self.photo_loss_handler = None

        self.photo_loss_handler = losses['photo'] if 'photo' in losses else None
        assert self.photo_loss_handler is not None, "You have to parametrize the photometric loss"

        self.n = self.photo_loss_handler.n

        self.smoothness_loss_handler = losses.get('smoothness', None)
        self.hinted_loss_handler = losses.get('hinted', None)

    def warp_ref_image(self, depth, ref_image, K, ref_K, pose):
        """
        Warps a reference image to produce a reconstruction of the original one.
        Parameters
        ----------
        inv_depths : torch.Tensor [B,1,DH,DW]
            Inverse depth map of the original image
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation
        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            Warped reference image (reconstructing the original one)
        """
        B, _, H, W = ref_image.shape
        device = ref_image.get_device()

        # Generate cameras
        _, _, DH, DW = depth.shape
        scale_factor = DW / float(W)
        cam = Camera(K=K.float()).scaled(scale_factor).to(device)
        ref_cam = Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device)

        ref_warped = view_synthesis(ref_image, depth, ref_cam, cam, padding_mode=self.photo_loss_handler.padding_mode)

        # Return warped reference image
        return ref_warped

    def warp_ref_images(self, depths, ref_image, K, ref_K, pose):
        """
        Warps a reference image using `warp_ref_image` to reconstructs a target image.
        This is done at each scale.

        Parameters
        ----------
        depths : torch.Tensor [B,1,H,W]
            Depth maps of the original image at all scales
        ref_image : torch.Tensor [B,3,H,W]
            Reference RGB image
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        ref_K : torch.Tensor [B,3,3]
            Reference camera intrinsics
        pose : Pose
            Original -> Reference camera transformation
        Returns
        -------
        ref_warped : torch.Tensor [B,3,H,W]
            List of Warped reference images (reconstructing the target view from source ones)
        """

        return [self.warp_ref_image(depths[i], ref_image, K, ref_K, pose) for i in range(self.n)]

    def reduce_loss(self, losses, name, reduce_op='min', mask=None, failure_masks=None):
        """
        Combine the loss from all context images
        Parameters
        ----------
        losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise losses from the entire context
        Returns
        -------
        reduced_loss : torch.Tensor [1]
            Reduced loss
        """

        # Reduce function
        def reduce_function(losses, failure_mask=None):
            if failure_mask is not None:
                cat = torch.cat(losses, 1)
                masked_cat = cat + 1000 * failure_mask
                argmins = masked_cat.argmin(1, True)
                out = cat.gather(1, argmins)
                if mask is not None:
                    out = out[mask]
                return out.mean()

            elif reduce_op == 'mean':
                return sum([l.mean() for l in losses]) / len(losses)
            elif reduce_op == 'min':
                if mask is None:
                    return torch.cat(losses, 1).min(1, True)[0].mean()
                else:
                    return torch.cat(losses, 1).min(1, True)[0][mask].mean()
            else:
                raise NotImplementedError(f'Unknown reduce_op: {reduce_op}')

        # Reduce photometric loss
        if failure_masks is not None:
            reduced_loss = sum([reduce_function(losses[i], failure_masks[i]) for i in range(self.n)]) / self.n
        else:
            reduced_loss = sum([reduce_function(losses[i]) for i in range(self.n)]) / self.n
        # Store and return reduced photometric loss
        self.add_metric(name, reduced_loss)
        return reduced_loss


    def forward(self, target_view, source_views, inv_depths, K, poses, gt_depth=None, failure_checks=None,
                progress=0.0):
        """
        Calculates training supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the source image, in all scales
        gt_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the source image
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """

        #self.n = self.progressive_scaling(progress)

        photometric_losses = [[] for _ in range(self.n)] # Container for losses computed with estimpated depth
        hinted_failure_masks=None
        failure_masks = None
        if failure_checks is not None:
            failure_masks = []
            hinted_failure_masks = []
            for i in range(self.n):
                b, _, h, w = inv_depths[i].shape
                n = len(poses)
                base_failure_mask = failure_checks.unsqueeze(-1).unsqueeze(-1).expand((b, n, h, w))
                copy = 1
                if self.photo_loss_handler.automask_loss:
                    copy += 1
                failure_mask = torch.repeat_interleave(base_failure_mask, copy, dim=1)
                # e,g: for len(poses) = 2
                # failure_mask: [photo1,automask1,photo2,automask2]
                failure_masks.append(failure_mask)
                if self.hinted_loss_handler is not None:
                    failure_mask = torch.cat([base_failure_mask, failure_mask], dim=1)
                    hinted_failure_masks.append(failure_mask)
                    # e,g: for len(poses) = 2
                    # hinted_failure_mask: [gt_photo1,gt_photo2,photo1,automask1,photo2,automask2]


        target_images = match_scales(target_view, inv_depths, self.n)
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]

        if self.hinted_loss_handler is not None:
            assert gt_depth is not None, "Ground Truth depth is required as input for the hinted loss"
            gt_photometric_losses = [[] for _ in range(self.n)] # Container for losses computed with GT depth
            gt_depths = match_scales(gt_depth, inv_depths, self.n)


        for (source_view, pose) in zip(source_views, poses):

            # Calculate warped images
            ref_warped = self.warp_ref_images(depths, source_view, K, K, pose)
            # Calculate and store image loss
            photometric_loss =  self.photo_loss_handler.calc_photometric_loss(ref_warped, target_images)
            for i in range(self.n):
                photometric_losses[i].append(photometric_loss[i])

            # If using automask
            if self.photo_loss_handler.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(source_view, inv_depths, self.n)
                unwarped_image_loss = self.photo_loss_handler.calc_photometric_loss(ref_images, target_images)
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[i])

            # If hinted loss required
            if self.hinted_loss_handler is not None:
                # Calculate warped images from get_depth
                ref_gt_warped = self.warp_ref_images(gt_depths, source_view, K, K, pose)
                # Calculate and store image loss
                gt_photometric_loss =  self.photo_loss_handler.calc_photometric_loss(ref_gt_warped, target_images)
                for i in range(self.n):
                    gt_depth_mask = (gt_depths[i] <= 0).float().detach()
                    # set loss for missing gt pixels to be high so they are never chosen as minimum
                    gt_photometric_losses[i].append(gt_photometric_loss[i] + 1000. * gt_depth_mask)

        # Calculate reduced loss
        mask = None
        if self.masked:
            assert gt_depth is not None, "Ground Truth depth is required as input to mask photo loss on LiDAR points"
            mask = (gt_depth > 0).detach()
        photo_loss = self.reduce_loss(photometric_losses, 'photometric_loss', mask=mask, failure_masks=failure_masks)

        # make a list as in-place sum is not auto-grad friendly
        losses = [photo_loss]

        if self.hinted_loss_handler is not None:
            depth_hints_loss = self.hinted_loss_handler(photometric_losses, gt_photometric_losses, inv_depths,
                                                        gt_depths, K, poses, failure_masks=hinted_failure_masks)
            losses.append(depth_hints_loss)
            self.merge_metrics(self.hinted_loss_handler)

        if self.smoothness_loss_handler is not None:
            smoothness_loss = self.smoothness_loss_handler(inv_depths, target_images)
            losses.append(smoothness_loss)
            self.merge_metrics(self.smoothness_loss_handler)

        total_loss = sum(losses)

        if torch.isnan(total_loss):
            total_loss = torch.zeros_like(total_loss, requires_grad=True)

            print('losses:', losses, file=sys.stderr)
            print('Nan error detected', file=sys.stderr)
            for i,inv_depth in enumerate(inv_depths):
                print(f'scale {i} mean_inv_depth:\n {inv_depth.mean(2, True).mean(3, True)}', file=sys.stderr)
                print(f'scale {i} inv_depth:\n {inv_depths}', file=sys.stderr)
            for i,pose in enumerate(poses):
                print('pose ',i, file=sys.stderr)
                print(pose.mat, file=sys.stderr)
            print('*'*30, file=sys.stderr)

        # Return losses and metrics
        return {
            'loss': total_loss.unsqueeze(0),
            'metrics': self.metrics,
        }


