# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.camera import Camera
from utils.multiview_warping_and_projection import view_synthesis
from utils.image import match_scales
from utils.depth import calc_smoothness, inv2depth
from losses.loss_base import LossBase, ProgressiveScaling


########################################################################################################################


def SSIM(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMlilarity (SSIM) distance between two images.
    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters
    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim


########################################################################################################################

def compute_color_gradients(input_img, kernelx, kernely):

    input_img = (input_img* 255).floor()

    sobelx_r = F.conv2d(input_img[:, 0, ...].unsqueeze(1), kernelx, padding=1)
    sobelx_g = F.conv2d(input_img[:, 1, ...].unsqueeze(1), kernelx, padding=1)
    sobelx_b = F.conv2d(input_img[:, 2, ...].unsqueeze(1), kernelx, padding=1)

    sobely_r = F.conv2d(input_img[:, 0, ...].unsqueeze(1), kernely, padding=1)
    sobely_g = F.conv2d(input_img[:, 1, ...].unsqueeze(1), kernely, padding=1)
    sobely_b = F.conv2d(input_img[:, 2, ...].unsqueeze(1), kernely, padding=1)


    sobelx = torch.cat([sobelx_r, sobelx_g, sobelx_b], 1)
    sobely = torch.cat([sobely_r, sobely_g, sobely_b], 1)

    #grad_magnitude = torch.sqrt(sobelx.pow(2) + sobely.pow(2))

    # convert to uint8 [0,255]
    #grad_magnitude = grad_magnitude.byte()

    # convert to uint8 [0,255]
    #sobelx = sobelx.byte()
    #sobely = sobely.byte()

    return sobelx, sobely


def get_sobel_kernels():
    sobel_kernelx = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobel_kernelx = torch.Tensor(sobel_kernelx).expand(1, 1, 3, 3)

    sobel_kernely = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    sobel_kernely = torch.Tensor(sobel_kernely).expand(1, 1, 3, 3)

    return sobel_kernelx, sobel_kernely

########################################################################################################################

class MultiViewPhotometricLoss(LossBase):
    """
    Self-Supervised multiview photometric loss.
    It takes two images, a depth map and a pose transformation to produce a
    reconstruction of one image from the perspective of the other, and calculates
    the difference between them
    Parameters
    ----------
    num_scales : int
        Number of inverse depth map scales to consider
    ssim_loss_weight : float
        Weight for the SSIM loss
    occ_reg_weight : float
        Weight for the occlusion regularization loss
    smooth_loss_weight : float
        Weight for the smoothness loss
    C1,C2 : float
        SSIM parameters
    photometric_reduce_op : str
        Method to reduce the photometric loss
    disp_norm : bool
        True if inverse depth is normalized for
    clip_loss : float
        Threshold for photometric loss clipping
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    padding_mode : str
        Padding mode for view synthesis
    automask_loss : bool
        True if automasking is enabled for the photometric loss
    kwargs : dict
        Extra parameters
    """

    def __init__(self, num_scales=4, ssim_loss_weight=0.85, occ_reg_weight=0.1, smooth_loss_weight=0.1,
                 uniformity_threshold=0.05, uniformity_weight=0.0, C1=1e-4, C2=9e-4, photometric_reduce_op='mean',
                 disp_norm=True, clip_loss=0.5, progressive_scaling=0.0, padding_mode='zeros', automask_loss=False,
                 laplace_loss=False):
        super().__init__()
        self.n = num_scales
        self.ssim_loss_weight = ssim_loss_weight
        self.occ_reg_weight = occ_reg_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.uniformity_threshold = uniformity_threshold
        self.uniformity_weight = uniformity_weight
        self.C1 = C1
        self.C2 = C2
        self.photometric_reduce_op = photometric_reduce_op
        self.disp_norm = disp_norm
        self.clip_loss = clip_loss
        self.padding_mode = padding_mode
        self.automask_loss = automask_loss
        self.laplace_loss = laplace_loss
        self.progressive_scaling = ProgressiveScaling(progressive_scaling, self.n)

        if self.laplace_loss:
            self.sobel_kernelx, self.sobel_kernely = get_sobel_kernels()

        # Asserts
        if self.automask_loss:
            assert self.photometric_reduce_op == 'min', \
                'For automasking only the min photometric_reduce_op is supported.'

    ########################################################################################################################

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'num_scales': self.n,
        }

    ########################################################################################################################


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

        ref_warped = view_synthesis(ref_image, depth, ref_cam, cam, padding_mode=self.padding_mode)

        # Return warped reference image
        return ref_warped

    def warp_ref_images(self, depths, ref_image, K, ref_K, pose):
        """
        Warps a reference image to produce a reconstruction of the original one.
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

    ########################################################################################################################

    def SSIM(self, x, y, kernel_size=3):
        """
        Calculates the SSIM (Structural SIMilarity) loss
        Parameters
        ----------
        x,y : torch.Tensor [B,3,H,W]
            Input images
        kernel_size : int
            Convolutional parameter
        Returns
        -------
        ssim : torch.Tensor [1]
            SSIM loss
        """
        ssim_value = SSIM(x, y, C1=self.C1, C2=self.C2, kernel_size=kernel_size)
        return torch.clamp((1. - ssim_value) / 2., 0., 1.)

    def calc_photometric_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales
        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i])
                   for i in range(self.n)]
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.SSIM(t_est[i], images[i], kernel_size=3)
                         for i in range(self.n)]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [self.ssim_loss_weight * ssim_loss[i].mean(1, True) +
                                (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                                for i in range(self.n)]
        else:
            photometric_loss = l1_loss
        # Clip loss
        if self.clip_loss > 0.0:
            for i in range(self.n):
                mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
                photometric_loss[i] = torch.clamp(
                    photometric_loss[i], max=float(mean + self.clip_loss * std))
        # Return total photometric loss
        return photometric_loss

    def reduce_loss(self, losses, name='photometric_loss', mask=None):
        """
        Combine the photometric loss from all context images
        Parameters
        ----------
        losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context
        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        """

        # Reduce function
        def reduce_function(losses):
            if self.photometric_reduce_op == 'mean':
                return sum([l.mean() for l in losses]) / len(losses)
            elif self.photometric_reduce_op == 'min':
                if mask is None:
                    return torch.cat(losses, 1).min(1, True)[0].mean()
                else:
                    return torch.cat(losses, 1).min(1, True)[0][mask].mean()
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))

        # Reduce photometric loss
        reduced_loss = sum([reduce_function(losses[i]) for i in range(self.n)]) / self.n
        # Store and return reduced photometric loss
        self.add_metric(name, reduced_loss)
        return reduced_loss

    ########################################################################################################################

    def calc_laplacian_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales
        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """

        laplacian_losses = []

        for i in range(self.n):
            target_sobelx, target_sobely = compute_color_gradients(images[i], self.sobel_kernelx, self.sobel_kernely)
            warped_sobelx, warped_sobely = compute_color_gradients(t_est[i], self.sobel_kernelx, self.sobel_kernely)

            #sqrd_err = (target_sobelx - warped_sobelx).pow(2) + (target_sobely - warped_sobely).pow(2)
            #err = torch.sqrt(sqrd_err)
            err = torch.abs(target_sobelx - warped_sobelx) + torch.abs(target_sobely - warped_sobely)
            err = err.sum(1, True)
            laplacian_losses.append(err)

            # Return total photometric loss
        return laplacian_losses

    ########################################################################################################################

    def calc_smoothness_loss(self, inv_depths, images):
        """
        Calculates the smoothness loss for inverse depth maps.
        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales
        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        # Calculate smoothness loss
        smoothness_loss = sum([(smoothness_x[i].abs().mean() +
                                smoothness_y[i].abs().mean()) / 2 ** i
                               for i in range(self.n)]) / self.n

        # Apply smoothness loss weight
        weighted_smoothness_loss = self.smooth_loss_weight * smoothness_loss

        # Store and return smoothness loss
        self.add_metric('smoothness_loss', weighted_smoothness_loss)
        return weighted_smoothness_loss

    def calc_uniformity_regularization(self, inv_depths):
        """
        Calculates the smoothness loss for inverse depth maps.
        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales
        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """

        # Calculate uniformity regulariztion
        uniformity_losses = []
        for i in range(self.n):
            inv_depth_mean = inv_depths[i].mean()
            squared_inv_depth = inv_depths[i] * inv_depths[i]
            variance = squared_inv_depth.mean() - inv_depth_mean*inv_depth_mean
            uniformity_losses.append(1/(self.uniformity_threshold*variance + variance*variance + 1e-6))

        # Apply smoothness loss weight
        uniformity_loss = self.uniformity_weight * sum(uniformity_losses) / self.n

        # Store and return uniformity loss
        self.add_metric('pred_variance', variance)
        self.add_metric('uniformity_loss', uniformity_loss)
        return uniformity_loss

    ########################################################################################################################

    def forward(self, target_view, source_views, inv_depths, K, poses, progress=0.0, mask=None):
        """
        Calculates training photometric loss.
        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        context : list of torch.Tensor [B,3,H,W]
            Context containing a list of reference images
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        K : torch.Tensor [B,3,3]
            Original camera intrinsics
        poses : list of Pose
            Camera transformation between original and context
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

        if self.laplace_loss:
            laplacian_losses = [[] for _ in range(self.n)]
            self.sobel_kernelx  = self.sobel_kernelx.type_as(target_view)
            self.sobel_kernely = self.sobel_kernely.type_as(target_view)

        target_images = match_scales(target_view, inv_depths, self.n)
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]

        # Loop over all reference images
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

        # Calculate reduced photometric loss
        total_photo_loss = self.reduce_loss(photometric_losses, mask=mask)

        losses = [total_photo_loss]

        if self.laplace_loss:
            total_laplacian_loss = self.reduce_loss(laplacian_losses, name='laplacian_loss', mask=mask)
            losses.append(total_laplacian_loss)

        # Include smoothness loss if requested
        if self.smooth_loss_weight > 0.0:
            smoothness_loss = self.calc_smoothness_loss(inv_depths, target_images)
            losses.append(smoothness_loss)

        # Include uniformity regularization loss if requested
        if self.uniformity_weight > 0.0:
            uniformity_loss = self.calc_uniformity_regularization(inv_depths)
            losses.append(uniformity_loss)

        total_loss = sum(losses)

        # Return losses and metrics
        return {
            'loss': total_loss.unsqueeze(0),
            'metrics': self.metrics,
        }

########################################################################################################################

