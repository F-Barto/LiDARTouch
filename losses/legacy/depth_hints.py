# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch

from utils.image import match_scales
from utils.depth import inv2depth

from losses.elements.supervised_loss import ReprojectedLoss, SupervisedLoss
from losses.legacy.multiview_photometric_loss import MultiViewPhotometricLoss
from losses.legacy.global_scaling import GlobalScalingLoss

class HintedMultiViewPhotometricLoss(MultiViewPhotometricLoss):

    """
    Semi-Supervised loss for inverse depth maps.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """

    def __init__(self, supervised_method='reprojected', hinted_loss_weight=1.0, supervised_num_scales=4,
                 global_scaling=False, **kwargs):
        super().__init__(**kwargs)

        self.global_scaling = global_scaling

        if self.global_scaling:
            self._global_scaling_loss = GlobalScalingLoss(supervised_num_scales)



        self.hinted_loss_weight = hinted_loss_weight

        if supervised_method == 'reprojected':
            self._supervised_loss = ReprojectedLoss(supervised_num_scales=supervised_num_scales, **kwargs)
            self.supervised_loss = self._supervised_loss.calculate_reprojected_losses
        else:
            self._supervised_loss = SupervisedLoss(supervised_method=supervised_method,
                                                   supervised_num_scales=supervised_num_scales,  **kwargs)
            self.supervised_loss = self._supervised_loss.calculate_losses

        self.supervised_method = supervised_method


    def calc_depth_hints_mask(self, photometric_losses, gt_photometric_losses):
        depth_hints_masks = []

        len_photo = len(photometric_losses[0]) # same for all scales
        len_gt_photo = len(gt_photometric_losses[0])  # same for all scales

        for i in range(self.n):
            all_losses = torch.cat(photometric_losses[i] + gt_photometric_losses[i], dim=1)

            # we keep all
            idxs = torch.argmin(all_losses, dim=1, keepdim=True).detach()


            # compute mask for each source views
            depth_hint_mask = []
            for i in range(len_photo, len_photo+len_gt_photo):
                depth_hint_mask.append((idxs == 2))
            depth_hint_mask = torch.cat(depth_hint_mask, dim=1)

            # if, in any source view, depth hint reprojection better than estimated or identity reprojection keep it
            depth_hint_mask = depth_hint_mask.any(dim=1, keepdim=True)

            depth_hints_masks.append(depth_hint_mask)

        return depth_hints_masks

    def calc_depth_hints_loss(self, depth_hints_masks, inv_depths, gt_depths, K, pose, progress=0.0):

        if self.supervised_method == 'reprojected':
            depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
            supervised_losses = self.supervised_loss(depths, gt_depths, K, pose, valid_masks=depth_hints_masks,
                                                     progress=progress)
        else:
            supervised_losses = self.supervised_loss(inv_depths, gt_depths, valid_masks=depth_hints_masks)

        depth_hints_loss = sum([supervised_losses[i].mean() for i in range(self.n)]) / self.n

        # Store and return reduced photometric loss
        self.add_metric('depth_hints_loss', depth_hints_loss)
        return depth_hints_loss


    def forward(self, target_view, source_views, inv_depths, gt_depth, K, poses, progress=0.0):
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

        # If using progressive scaling
        self.n = self.progressive_scaling(progress)

        photometric_losses = [[] for _ in range(self.n)]
        gt_photometric_losses = [[] for _ in range(self.n)]

        target_images = match_scales(target_view, inv_depths, self.n)
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        gt_depths = match_scales(gt_depth, depths, self.n)

        if self.laplace_loss:
            laplacian_losses = [[] for _ in range(self.n)]
            self.sobel_kernelx  = self.sobel_kernelx.type_as(target_view)
            self.sobel_kernely = self.sobel_kernely.type_as(target_view)

        for (source_view, pose) in zip(source_views, poses):

            # Calculate warped images
            ref_warped = self.warp_ref_images(depths, source_view, K, K, pose)
            # Calculate and store image loss
            photometric_loss = self.calc_photometric_loss(ref_warped, target_images)
            for i in range(self.n):
                photometric_losses[i].append(photometric_loss[i])

            if self.laplace_loss:
                laplacian_loss = self.calc_laplacian_loss(ref_warped, target_images)
                for i in range(self.n):
                    laplacian_losses[i].append(laplacian_loss[i])

            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                ref_images = match_scales(source_view, inv_depths, self.n)
                unwarped_image_loss = self.calc_photometric_loss(ref_images, target_images)
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[i])

                if self.laplace_loss:
                    unwarped_laplacian_loss = self.calc_laplacian_loss(ref_images, target_images)
                    for i in range(self.n):
                        laplacian_losses[i].append(unwarped_laplacian_loss[i])

            # Calculate warped images from get_depth
            ref_gt_warped = self.warp_ref_images(gt_depths, source_view, K, K, pose)
            # Calculate and store image loss
            gt_photometric_loss = self.calc_photometric_loss(ref_gt_warped, target_images)
            for i in range(self.n):
                gt_depth_mask = (gt_depths[i] <= 0).float().detach()
                # set loss for missing gt pixels to be high so they are never chosen as minimum
                gt_photometric_losses[i].append(gt_photometric_loss[i] + 1000. * gt_depth_mask)

        # Calculate reduced loss
        loss = self.reduce_loss(photometric_losses)

        depth_hints_mask = self.calc_depth_hints_mask(photometric_losses, gt_photometric_losses)

        # here the argument "pose" is only used if we use the reprojection loss from toyota
        # which source wiew is used is not important, read more at https://arxiv.org/abs/1910.01765
        depth_hints_loss = self.calc_depth_hints_loss(depth_hints_mask, inv_depths, gt_depths, K, poses[0], progress=progress)
        depth_hints_loss = self.hinted_loss_weight * depth_hints_loss

        # make a list as in-pace sum is not auto-grad friendly
        losses = [loss, depth_hints_loss]

        if self.global_scaling:
            global_scaling_outputs = self._global_scaling_loss(inv_depths, gt_depth)

            losses.append(global_scaling_outputs['loss'])
            self.metrics.update(global_scaling_outputs['metrics'])

        if self.laplace_loss:
            total_laplacian_loss = self.reduce_loss(laplacian_losses, name='laplacian_loss')
            losses.append(total_laplacian_loss)

        # Include smoothness loss if requested
        if self.smooth_loss_weight > 0.0:
            losses.append(self.calc_smoothness_loss(inv_depths, target_images))

        # Include uniformity regularization loss if requested
        if self.uniformity_weight > 0.0:
            losses.append(self.calc_uniformity_regularization(inv_depths))

        total_loss = sum(losses)

        # Return losses and metrics
        return {
            'loss': total_loss.unsqueeze(0),
            'metrics': self.metrics,
        }


