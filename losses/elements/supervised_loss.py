# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F



from utils.image import match_scales
from utils.camera import Camera
from utils.multiview_warping_and_projection import reconstruct
from losses.loss_base import LossBase, ProgressiveScaling
from utils.depth import inv2depth, depth2inv
########################################################################################################################

class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        """
        Initializes the BerHuLoss class.

        Parameters
        ----------
        threshold : float
            Mask parameter
        """
        super().__init__()
        self.threshold = threshold
    def forward(self, pred, gt):
        """
        Calculates the BerHu loss.

        Parameters
        ----------
        pred : torch.Tensor [B,1,H,W]
            Predicted inverse depth map
        gt : torch.Tensor [B,1,H,W]
            Ground-truth inverse depth map

        Returns
        -------
        loss : torch.Tensor [1]
            BerHu loss
        """
        huber_c = torch.max(pred - gt)
        huber_c = self.threshold * huber_c
        diff = (pred - gt).abs()

        # Remove
        # mask = (gt > 0).detach()
        # diff = gt - pred
        # diff = diff[mask]
        # diff = diff.abs()

        huber_mask = (diff > huber_c).detach()
        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2
        return torch.cat((diff, diff2)).mean()

class SilogLoss(nn.Module):
    def __init__(self, ratio=10, ratio2=0.85):
        super().__init__()
        self.ratio = ratio
        self.ratio2 = ratio2

    def forward(self, pred, gt):
        log_diff = torch.log(pred * self.ratio) - \
                   torch.log(gt * self.ratio)
        silog1 = torch.mean(log_diff ** 2)
        silog2 = self.ratio2 * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * self.ratio
        return silog_loss

########################################################################################################################

def get_loss_func(supervised_method):
    """Determines the supervised loss to be used, given the supervised method."""
    if supervised_method.endswith('l1'):
        return nn.L1Loss()
    elif supervised_method.endswith('mse'):
        return nn.MSELoss()
    elif supervised_method.endswith('berhu'):
        return BerHuLoss()
    elif supervised_method.endswith('silog'):
        return SilogLoss()
    elif supervised_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    else:
        raise ValueError('Unknown supervised loss {}'.format(supervised_method))

########################################################################################################################

class SupervisedLoss(LossBase):
    """
    Supervised loss for inverse depth maps.

    Parameters
    ----------
    supervised_method : str
        Which supervised method will be used
    supervised_num_scales : int
        Number of scales used by the supervised loss
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_method='sparse-l1',
                 supervised_num_scales=4, supervised_loss_weight=1.0, progressive_scaling=0.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_func = get_loss_func(supervised_method)
        self.supervised_method = supervised_method
        self.n = supervised_num_scales
        self.progressive_scaling = ProgressiveScaling(progressive_scaling, self.n)
        self.supervised_loss_weight = supervised_loss_weight


    def calculate_losses(self, inv_depths, gt_depths, valid_masks=None):
        """
        Calculate the supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : list of torch.Tensor [B,1,H,W]
            List of ground-truth inverse depth maps

        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """

        losses = []

        gt_inv_depths = [depth2inv(gt_depths[i]) for i in range(self.n)]

        # If using a sparse loss, mask invalid pixels for all scales
        if self.supervised_method.startswith('sparse'):
            for i in range(self.n):
                mask = (gt_depths[i] > 0.).detach()

                if valid_masks is not None:
                    mask = mask & valid_masks[i]

                masked_inv_depth = inv_depths[i][mask]
                masked_gt_inv_depth = gt_inv_depths[i][mask]

                loss = self.loss_func(masked_inv_depth, masked_gt_inv_depth)
                losses.append(loss)


        # Return per-scale average loss
        return losses

    def forward(self, inv_depths, gt_depth, progress=0.0):
        """
        Calculates training supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the original image
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """
        # If using progressive scaling
        self.n = self.progressive_scaling(progress)

        # Match predicted scales for ground-truth
        gt_depths = match_scales(gt_depth, inv_depths, self.n, mode='nearest')

        # Calculate and store supervised loss
        losses = self.calculate_losses(inv_depths, gt_depths)
        loss = sum([losses[i] for i in range(self.n)]) / self.n

        self.add_metric('supervised_loss', loss)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0) * self.supervised_loss_weight,
            'metrics': self.metrics,
        }


##########################################################################

# Uses project & reconstruct from utils.multiview_warping_and_projection
# how to handle masking ? pixel falling out of bounds ?

class ReprojectedLoss(LossBase):

    """
    Supervised loss for inverse depth maps.

    Parameters
    ----------
    supervised_method : str
        Which supervised method will be used
    supervised_num_scales : int
        Number of scales used by the supervised loss
    progressive_scaling : float
        Training percentage for progressive scaling (0.0 to disable)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, supervised_num_scales=4, progressive_scaling=0.0, **kwargs):
        super().__init__()
        self.n = supervised_num_scales
        self.progressive_scaling = ProgressiveScaling(
            progressive_scaling, self.n)

    @property
    def logs(self):
        """Returns class logs."""
        return {
            'supervised_num_scales': self.n,
        }

    def coords_projected(self, camera, depth):

        B, C, H, W = depth.shape
        assert C == 1

        world_3d_points = reconstruct(camera, depth, frame='w')

        # Project world points onto source camera
        camera_3d_points = camera.K.bmm((camera.Tcw @ world_3d_points).view(B, 3, -1))

        Z = camera_3d_points[:,2].clamp(min=1e-5).unsqueeze(1)

        camera_2d_points = camera_3d_points[:, :2] / Z.repeat([1, 2, 1])

        return camera_2d_points

    def calculate_reprojected_loss(self, depth, gt_depth, K, pose, valid_mask=None):
        """
        Calculate the supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            List of predicted inverse depth maps
        gt_inv_depths : torch.Tensor [B,1,H,W]
            ground-truth inverse depth map

        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """

        B, _, H, W = gt_depth.shape
        device = gt_depth.get_device()

        # Generate camera at scale
        B, _, DH, DW = depth.shape
        scale_factor = DW / float(W)
        # we project into source view camera, hence the Tcw=pose
        cam = Camera(K=K.float(), Tcw=pose).scaled(scale_factor).to(device)

        # Get the reprojected pixels coordinate from estimated depth
        coords_pred_depth = self.coords_projected(cam, depth)
        coords_gt_depth   = self.coords_projected(cam, gt_depth)

        # mask invalid pixels for all scales
        mask = (gt_depth > 0.).detach() # don't need to compute grad for mask

        if valid_mask is not None:
            mask = mask & valid_mask

        mask = mask.view(B,1,-1).repeat([1,2,1]).detach()

        coords_pred_depth = coords_pred_depth[mask].view(2,-1) # applying mask flatten [x0,...,y0,...] so apply view
        coords_gt_depth = coords_gt_depth[mask].view(2,-1)
        loss = F.pairwise_distance(coords_pred_depth, coords_gt_depth, p=2.0, eps=1e-06, keepdim=False)

        # Return reprojected loss
        return loss

    def calculate_reprojected_losses(self, inv_depths, gt_depths, K, poses, valid_masks=None, progress=0.0):
        """
        Calculate the supervised loss.

        Parameters
        ----------
        depths : list of torch.Tensor [B,1,H,W]
            List of predicted depth maps
        gt_depths : torch.Tensor [B,1,H,W]
            ground-truth depth maps

        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """

        # If using progressive scaling
        self.n = self.progressive_scaling(progress)

        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]

        reprojected_losses = [[] for _ in range(self.n)]
        for pose in poses:
            # Calculate and store supervised loss for each scale
            for i in range(self.n):
                # Get the reprojected pixels coordinate from estimated depth
                valid_mask = valid_masks[i] if valid_masks is not None else None
                loss = self.calculate_reprojected_loss(depths[i], gt_depths[i], K, pose, valid_mask=valid_mask)
                reprojected_losses[i].append(loss.unsqueeze(0))

        # min across poses
        reprojected_losses = [torch.cat(reprojected_losses[i], 0).min(0, True)[0].mean() for i in range(self.n)]

        return reprojected_losses


    def forward(self, inv_depths, gt_depth, K, poses, valid_masks=None, progress=0.0):
        """
        Calculates training supervised loss.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted depth maps for the original image, in all scales
        gt_inv_depth : torch.Tensor [B,1,H,W]
            Ground-truth depth map for the original image
        return_logs : bool
            True if logs are saved for visualization
        progress : float
            Training percentage

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """

        # NOTE/TODO: here we compute loss for each pose but it is useless as no rgb image is used ---> is it really ?
        # just take one pose (e.g., the first one)
        gt_depths = match_scales(gt_depth, inv_depths, self.n)

        reprojected_losses = self.calculate_reprojected_losses(inv_depths, gt_depths, K, poses,
                                                               valid_masks=valid_masks, progress=progress)

        # Return per-scale reprojected loss mean
        reprojected_loss = sum(reprojected_losses) / self.n

        self.add_metric('reprojected_loss', reprojected_loss)
        # Return losses and metrics
        return {
            'loss': reprojected_loss.unsqueeze(0),
            'metrics': self.metrics,
        }




#########################################################################################

class MultimodalSelfTeachingLoss(LossBase):
    """
    Supervised loss for inverse depth maps.

    Parameters
    ----------

    """
    def __init__(self, supervised_num_scales=4):
        super().__init__()

        self.n = supervised_num_scales

    def calculate_losses(self, teacher_preds, student_preds, student_logvars):
        """
        Calculate the supervised loss.

        Returns
        -------
        loss : torch.Tensor [1]
            Average supervised loss for all scales
        """

        losses = []

        for i in range(self.n):
            # "From What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NIPS 2017
            # In practice, we train the network to predict the log variance
            loss1 = torch.exp(-student_logvars[i]) * torch.square((teacher_preds[i] - student_preds[i]))
            loss1 = loss1.mean((2,3), keepdim=True)
            loss2 = student_logvars[i].mean((2,3), keepdim=True)

            loss = .5 * (loss1 + loss2)

            losses.append(loss)

        # Return per-scale average loss
        return losses

    def forward(self, multimodal_teachers_preds, student_preds, student_logvars, weights=None):
        """
        Calculates training supervised loss.

        Parameters
        ----------

        Returns
        -------
        losses_and_metrics : dict
            Output dictionary
        """

        nb_teachers = len(multimodal_teachers_preds)
        B,_,_,_ = student_preds[0].shape

        teachers_losses = [[] for _ in range(self.n)]
        for teacher_pred in multimodal_teachers_preds:
            teacher_preds = match_scales(teacher_pred, student_preds, self.n)
            losses = self.calculate_losses(teacher_preds, student_preds, student_logvars)
            for i in range(self.n):
                teachers_losses[i].append(losses[i])

        teachers_losses = [torch.cat(teachers_loss, 1) for teachers_loss in teachers_losses] # B x nb_teachers x 1 x 1

        if weights is None:
            weights = torch.ones(B,nb_teachers,1,1).as_type(student_preds[0])

        teachers_losses = [(teachers_loss * weights).sum(1).mean() for teachers_loss in teachers_losses] # weigthed sum

        loss = sum(teachers_losses) / self.n


        self.add_metric('multimodal_selfteaching_loss', loss)
        # Return losses and metrics
        return {
            'loss': loss.unsqueeze(0),
            'metrics': self.metrics,
        }