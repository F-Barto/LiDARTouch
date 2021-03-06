import torch

from losses.elements.supervised_loss import SupervisedLoss, ReprojectedLoss
from losses.loss_base import LossBase


class HintedLoss(LossBase):
    def __init__(self, supervised_method='reprojected', hinted_loss_weight=1.0, supervised_num_scales=4):
        super().__init__()

        self.hinted_loss_weight = hinted_loss_weight

        self.supervised_method = supervised_method

        if self.supervised_method == 'reprojected':
            _supervised_loss = ReprojectedLoss(supervised_num_scales=supervised_num_scales)
            self.supervised_loss = _supervised_loss.calculate_reprojected_losses
        else:
            _supervised_loss = SupervisedLoss(supervised_method=supervised_method,
                                                   supervised_num_scales=supervised_num_scales)
            self.supervised_loss = _supervised_loss.calculate_losses

        self.supervised_method = supervised_method

        self.n = supervised_num_scales


    def calc_depth_hints_mask(self, photometric_losses, gt_photometric_losses, failure_masks=None):
        depth_hints_masks = []

        len_gt_photo = len(gt_photometric_losses[0]) # the length is the same for all scales, hence just [0]

        for i in range(self.n):

            # concat photo loss from pred depth, automask ans gt LiDAR
            all_losses = torch.cat(gt_photometric_losses[i] + photometric_losses[i], dim=1)

            # check which depth in which source view produce lower photo loss: estimated, automasked, or LiDAR
            if failure_masks is not None:
                failure_mask = failure_masks[i]
                # set a slightly less value for gt_photometric_losses mask so that if all poses are failed
                # gt_photometric index is selected and we have regression loss
                # emulate <= instead of just <)
                failure_mask[:,:len_gt_photo,:,:] = failure_mask[:,:len_gt_photo,:,:] * 400
                failure_mask[:, len_gt_photo:, :, :] = failure_mask[:, len_gt_photo:, :, :] * 500
                all_losses += failure_mask
            idxs = torch.argmin(all_losses, dim=1, keepdim=True).detach()

            # check for valid depth hint in each source view  (photo loss min for depth from LiDAR)
            depth_hint_mask = []
            for j in range(len_gt_photo):
                depth_hint_mask.append((idxs == j))

            # if, in any source view, depth hint reprojection better than estimated depth reprojection keep it
            depth_hint_mask = torch.cat(depth_hint_mask, dim=1)
            depth_hint_mask = depth_hint_mask.any(dim=1, keepdim=True)

            depth_hints_masks.append(depth_hint_mask)

        return depth_hints_masks


    def calc_depth_hints_loss(self, *args, **kwargs):

        supervised_losses = self.supervised_loss(*args, **kwargs)

        depth_hints_loss = sum([supervised_losses[i] for i in range(self.n)]) / self.n

        # Store and return reduced photometric loss
        self.add_metric('depth_hints_loss', depth_hints_loss)
        return depth_hints_loss

    def forward(self, photometric_losses, gt_photometric_losses, inv_depths, gt_depths, K, poses, failure_masks=None):

        depth_hints_mask = self.calc_depth_hints_mask(photometric_losses, gt_photometric_losses,
                                                      failure_masks=failure_masks)

        if self.supervised_method == 'reprojected':
            args = (inv_depths, gt_depths, K, poses)
        else:
            args = (inv_depths, gt_depths)

        depth_hints_loss = self.calc_depth_hints_loss(*args, valid_masks=depth_hints_mask)
        depth_hints_loss = self.hinted_loss_weight * depth_hints_loss

        return depth_hints_loss