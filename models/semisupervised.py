'''
This module houses:

Model definition (init)
Computations (forward)
What happens inside the training loop (training_step)
What happens inside the validation loop (validation_step)
What optimizer(s) to use (configure_optimizers)
What data to use (train_dataloader, val_dataloader, test_dataloader)

'''

import random
import copy

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader  import default_collate

import pytorch_lightning as pl
from pytorch_lightning import _logger as terminal_logger
from pytorch_lightning.core.decorators import auto_move_data

from networks.legacy.packnet.packnet import PackNet01
from networks.legacy.packnet.posenet import PoseNet
from networks.legacy.monodepth2.depth_rest_net import DepthResNet
from networks.legacy.monodepth2.pose_res_net import PoseResNet
from networks.legacy.monodepth2.guided_depth_rest_net import GuidedDepthResNet
from networks.legacy.monodepth2.teacher_guided_depth_rest_net import TeacherGuidedDepthResNet
from networks.legacy.custom.guided_sparse_dilated_depth_net import GuidedSparseDepthResNet

from networks.legacy.monodepth_original.depth_res_net import DepthResNet as OriginalDepthResNet
from networks.legacy.monodepth_original.pose_res_net import PoseResNet as OriginalPoseResNet

from losses.legacy.multiview_photometric_loss import MultiViewPhotometricLoss
from losses.elements.supervised_loss import SupervisedLoss, MultimodalSelfTeachingLoss
from losses.elements.velocity_loss import VelocityLoss
from losses.legacy.depth_hints import HintedMultiViewPhotometricLoss

from dataloaders.kitti import SequentialKittiLoader
from dataloaders.randcam_argoverse import RandCamSequentialArgoverseLoader
from dataloaders.transforms import train_transforms, val_transforms, test_transforms

from utils.pose import Pose
from utils.image import interpolate_scales, flip_lr
from utils.depth import inv2depth, compute_depth_metrics

from utils.common_logging import average_metrics
from utils.wandb_logging import prepare_images_to_log as wandb_prep_images
from utils.tensorboard_logging import prepare_images_to_log as tensorboard_prep_images
from utils.loading import load_tri_network
from utils.misc import make_list

from .loss_handlers import self_teaching_loss_handler


IMPLEMENTED_ROTATION_MODES = ['euler']
TENSORBOARD_LOGGER_KEY = 'tensorboard'
WANDB_LOGGER_KEY = 'wandb'

def prepare_data(datasets_config, input_channels=3):
    terminal_logger.info("Preparing Datasets...")

    if datasets_config.dataset_name == 'rand_cam_argoverse':
        dataset_cls = RandCamSequentialArgoverseLoader
    elif datasets_config.dataset_name == 'kitti':
        dataset_cls = SequentialKittiLoader
    else:
        raise ValueError(f'Dataset of class {datasets_config.dataset_name} is not implemented')

    train_dataset = dataset_cls(**datasets_config.train, data_transform=train_transforms,
                                          input_channels=input_channels)
    print('len train_dataset: ', len(train_dataset))

    val_dataset = dataset_cls(**datasets_config.val, data_transform=val_transforms,
                                          input_channels=input_channels)
    print('len val_dataset: ', len(val_dataset))

    test_dataset = dataset_cls(**datasets_config.test, data_transform=test_transforms,
                                          input_channels=input_channels)
    print('len test_dataset: ', len(test_dataset))

    return train_dataset, val_dataset, test_dataset

class MonocularSemiSupDepth(pl.LightningModule):
    def __init__(self, hparams,):
        super().__init__()


        self.hparams = hparams

        train_dataset, val_dataset, test_dataset = prepare_data(hparams.datasets, hparams.input_channels)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.input_channels = hparams.input_channels

        ################### HPARAMS Validation #####################

        assert hparams.rotation_mode in IMPLEMENTED_ROTATION_MODES, \
            f"Option `rotation_mode` should be in {IMPLEMENTED_ROTATION_MODES}"
        self.rotation_mode = hparams.rotation_mode

        assert self.hparams.logger in [TENSORBOARD_LOGGER_KEY, WANDB_LOGGER_KEY], \
            f"Logger should either be {TENSORBOARD_LOGGER_KEY} or {WANDB_LOGGER_KEY}"


        ################### Model Definition #####################

        # Depth Net
        if self.hparams.model.depth_net.name == 'packnet':
            self.depth_net = PackNet01(**hparams.model.depth_net.options, input_channels=self.input_channels)
        elif self.hparams.model.depth_net.name == 'monodepth':
            self.depth_net = DepthResNet(**hparams.model.depth_net.options, input_channels=self.input_channels)
        elif self.hparams.model.depth_net.name == 'monodepth_original':
            self.depth_net = OriginalDepthResNet(**hparams.model.depth_net.options)
        elif self.hparams.model.depth_net.name == 'guiding':
            assert train_dataset.load_sparse_depth, "Sparse depth signal is necessary for feature guidance."
            self.depth_net = GuidedDepthResNet(**hparams.model.depth_net.options, input_channels=self.input_channels)
        elif self.hparams.model.depth_net.name == 'sparse-guiding':
            assert train_dataset.load_sparse_depth, "Sparse depth signal is necessary for feature guidance."
            self.depth_net = GuidedSparseDepthResNet(**hparams.model.depth_net.options, input_channels=self.input_channels)
        elif self.hparams.model.depth_net.name == 'teacher-guiding':
            assert train_dataset.load_sparse_depth, "Sparse depth signal is necessary for feature guidance."
            self.depth_net = TeacherGuidedDepthResNet(**hparams.model.depth_net.options,
                                                      input_channels=self.input_channels)
        else:
            terminal_logger.error(f"Depth net {self.hparams.model.depth_net.name} not implemented")

        # Pose Net
        if hasattr(self.hparams.datasets.train, 'use_pnp') and self.hparams.datasets.train.use_pnp:
            self.pose_net = None
        elif self.hparams.model.pose_net.name == 'packnet':
            self.pose_net = PoseNet(**hparams.model.pose_net.options, input_channels=self.input_channels)
        elif self.hparams.model.pose_net.name == 'monodepth':
            self.pose_net = PoseResNet(**hparams.model.pose_net.options, input_channels=self.input_channels)
        elif self.hparams.model.pose_net.name == 'monodepth_original':
            self.pose_net = OriginalPoseResNet(**hparams.model.pose_net.options)
        else:
            terminal_logger.error(f"Pose net {self.hparams.model.pose_net.name} not implemented")

        ################### Checkpoint loading Definition #####################

        tri_checkpoint_path =  self.hparams.model.get('tri_checkpoint_path', None)
        if tri_checkpoint_path is not None:
            load_tri_network(self, tri_checkpoint_path)

        ################### Losses Definition #####################

        self.multi_scale_pred = True
        print('hasattr(self.depth_net, multi_scale): ', hasattr(self.depth_net, 'multi_scale'))
        if hasattr(self.depth_net, 'multi_scale'):
            self.multi_scale_pred =  self.depth_net.multi_scale

        self.hinted_supervision = False
        if self.hparams.losses.get('HintedMultiViewPhotometricLoss', None) is not None:
            self.hinted_supervision = True

            if not self.multi_scale_pred:
                self.hparams.losses.HintedMultiViewPhotometricLoss.supervised_num_scales = 1
                self.hparams.losses.HintedMultiViewPhotometricLoss.num_scales = 1

            self._hinted_loss = HintedMultiViewPhotometricLoss(**self.hparams.losses.HintedMultiViewPhotometricLoss)

        else:

            if not self.multi_scale_pred:
                self.hparams.losses.PhotometricLoss.num_scales = 1

            # Photometric loss used as main supervisory signal
            self.self_supervised_loss = MultiViewPhotometricLoss(**self.hparams.losses.PhotometricLoss)

            if self.hparams.losses.get('supervised_loss_weight', 0.0) > 0.0:

                if not self.multi_scale_pred:
                    self.hparams.losses.SupervisedLoss.num_scales = 1

                self.supervised_loss = SupervisedLoss(**self.hparams.losses.SupervisedLoss)
                self.supervised_loss_weight = hparams.losses.supervised_loss_weight

        if self.hparams.losses.get('velocity_loss_weight', 0.0) > 0.0:
            assert self.hparams.datasets.train.load_pose == True, 'GT translation magnitude is required for velocity loss.'
            self.velocity_supervision = True
            self._velocity_loss = VelocityLoss()
            self.velocity_loss_weight = hparams.losses.velocity_loss_weight
        else:
            assert self.hparams.datasets.train.load_pose == False, \
                'GT translation magnitude should not be load if no velocity supervision.'

        if self.hparams.model.depth_net.name == 'teacher-guiding':

            self._selfteaching_loss = MultimodalSelfTeachingLoss(4)

            monoscale_hinted_loss_params = copy.deepcopy(self.hparams.losses.HintedMultiViewPhotometricLoss)
            monoscale_hinted_loss_params.supervised_num_scales = 1
            monoscale_hinted_loss_params.num_scales = 1
            self._monoscale_hinted_loss = HintedMultiViewPhotometricLoss(**monoscale_hinted_loss_params)


    def compute_common_losses_and_metrics(self, batch, disp_preds, poses_preds, progress, metrics_prefix=''):
        losses = []
        metrics = {}

        if self.hinted_supervision:
            hinted_output = self._hinted_loss(
                batch['target_view_original'],
                batch['source_views_original'],
                disp_preds,
                batch['sparse_projected_lidar_original'],
                batch['intrinsics'],
                poses_preds,
                progress=progress)
            losses.append(hinted_output['loss'])
            metrics.update({metrics_prefix + k : v for k,v in hinted_output['metrics'].items()})

        else:

            mask = None
            if self.hparams.losses.get('masked_photo', False):
                mask = (batch['sparse_projected_lidar_original'] > 0).detach()

            self_sup_output = self.self_supervised_loss(
                batch['target_view_original'],
                batch['source_views_original'],
                disp_preds,
                batch['intrinsics'],
                poses_preds,
                progress=progress,
                mask = mask)

            losses.append(self_sup_output['loss'])
            metrics.update({metrics_prefix + k: v for k, v in self_sup_output['metrics'].items()})

            if self.hparams.losses.get('supervised_loss_weight', 0.0) > 0.0:
                # Calculate and weight supervised loss
                sup_output = self.supervised_loss(disp_preds, batch['sparse_projected_lidar_original'],
                                                  progress=progress)

                losses.append(self.supervised_loss_weight * sup_output['loss'])
                metrics.update({metrics_prefix + k: v for k, v in sup_output['metrics'].items()})

        translation_magnitudes = batch.get('translation_magnitudes', None)
        if translation_magnitudes is not None:
            velocity_output = self._velocity_loss(poses_preds, translation_magnitudes)

            losses.append(self.velocity_loss_weight * velocity_output['loss'])
            metrics.update({metrics_prefix + k: v for k, v in velocity_output['metrics'].items()})

        return losses, metrics

    def compute_inv_depths(self, image, sparse_depth=None, nn_diff_pts_3d=None, pixel_idxs=None, nn_pixel_idxs=None):
        """Computes inverse depth maps from single images"""

        flip = random.random() < 0.5 if self.training else False

        if flip:
            image = flip_lr(image)

        if sparse_depth is None:
            output = self.depth_net(image)
        elif nn_diff_pts_3d is not None:
            output = self.depth_net(image, sparse_depth, nn_diff_pts_3d, pixel_idxs, nn_pixel_idxs)
        else:
            output = self.depth_net(image, sparse_depth)

        if not isinstance(output, dict): # not self-teaching, the output is just a list containing inv_depths
            inv_depths = output
            inv_depths = make_list(inv_depths)

            if flip:
                inv_depths = [flip_lr(inv_depth) for inv_depth in inv_depths]

            if self.hparams.upsample_depth_maps:
                inv_depths = interpolate_scales(inv_depths, mode='nearest')

            # Return inverse depth maps
            return inv_depths

        else:

            keys = ['inv_depths', 'disp', 'coarse_disp', 'cam_disp', 'lidar_disp']

            if output.get('uncertainties', None) is not None:
                keys.append('uncertainties')

            for key in keys:

                if key not in output:
                    continue

                if flip:
                    output[key] = [flip_lr(o) for o in make_list(output[key])]
                else:
                    output[key] = make_list(output[key])

                # interpolate preds from all scales to biggest scale
                if key in ['inv_depths', 'uncertainties'] and  self.hparams.upsample_depth_maps:
                    output[key] = interpolate_scales(output[key], mode='nearest')
                    # 'cam_disp', 'disp', 'lidar_disp' are one scales preds so we don't have to interpolate

            return output


    def compute_poses(self, target_view, source_views):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.pose_net(target_view, source_views)
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode) for i in range(pose_vec.shape[1])]

    def scale_poses(self, poses, translation_magnitudes):

        # norms = [torch.norm(pose.translation, p=2, dim=1, keepdim=True) for pose in poses]
        #         return [Pose(pose.mat.clone()).translation / norm * translation_magnitude.unsqueeze(1)
        #                 for pose, translation_magnitude in zip(poses, translation_magnitudes)]

        scaled_poses = []
        for pose, translation_magnitude in zip(poses, translation_magnitudes):


            # clone otherwise raise gradient error
            scaled_pose = Pose(pose.mat.clone())
            norm =  torch.norm(pose.translation, p=2, dim=1, keepdim=True) + 1e-8

            scaled_pose.translation = pose.translation / norm * translation_magnitude.unsqueeze(1)
            scaled_poses.append(scaled_pose)

        return scaled_poses

    def evaluate_depth(self, batch):
        """
        Evaluate batch to produce depth metrics.

        Returns
        -------
        output : dict
            Dictionary containing a "metrics" and a "inv_depth" key

            metrics : torch.Tensor [7]
                Depth metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)

            inv_depth:
                predicted inverse depth
        """
        # Get predicted depth
        output = self(batch)

        output_key = 'disp' if not self.multi_scale_pred else 'inv_depths'

        inv_depth = output[output_key][0]
        depth = inv2depth(inv_depth)

        # Calculate predicted metrics
        metrics = compute_depth_metrics(gt=batch['projected_lidar'], pred=depth, **self.hparams.metrics)
        # Return metrics and extra information

        result = {
            'metrics': metrics,
            'inv_depth': inv_depth,
            'depth': depth,
        }

        for key in ['cam_disp', 'lidar_disp', 'uncertainties', 'coarse_disp']:
            if key in output:
                result[key] = output[key][0]

        return result

    @auto_move_data
    def forward(self, batch):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch

        Returns
        -------
        output : dict
            Dictionary containing predicted inverse depth maps and poses
        """

        if batch.get('nn_diff_pts_3d', None) is not None:
            output = self.compute_inv_depths(batch['target_view'],
                                                 sparse_depth=batch['sparse_projected_lidar'],
                                                 nn_diff_pts_3d=batch['nn_diff_pts_3d'],
                                                 pixel_idxs=batch['pixel_idxs'],
                                                 nn_pixel_idxs=batch['nn_pixel_idxs'])
        elif "guiding" in self.hparams.model.depth_net.name:
            output = self.compute_inv_depths(batch['target_view'], sparse_depth=batch['sparse_projected_lidar'])
        else:
            output = self.compute_inv_depths(batch['target_view'])

        poses = None
        if 'poses_pnp' in batch:
            pose_vec = batch['poses_pnp'].float()
            poses = [Pose.from_vec(pose_vec[:, i], self.rotation_mode) for i in range(pose_vec.shape[1])]
        elif 'source_views' in batch and self.pose_net is not None:
            poses = self.compute_poses(batch['target_view'], batch['source_views'])

            #translation_magnitudes = batch.get('translation_magnitudes', None)
            #if translation_magnitudes is not None:
            #    poses = self.scale_poses(poses, translation_magnitudes)
        else:
            pass

        preds = {
            'poses': poses,
        }

        if not isinstance(output, dict):
            preds['inv_depths'] = output # case we return inv_depths as list
        else:
            preds.update(output)

        if not self.training:
            return preds
        else:
            progress = self.current_epoch / self.hparams.trainer.max_epochs

            if not self.multi_scale_pred:

                losses, metrics = self.compute_common_losses_and_metrics(batch, preds['disp'], poses, progress)

                if 'coarse_disp' in preds:
                    coarse_losses, coarse_metrics = \
                        self.compute_common_losses_and_metrics(batch, preds['coarse_disp'], poses,
                                                               progress, metrics_prefix='coarse')

                    losses += coarse_losses
                    coarse_metrics.update(metrics)

            else:
                losses, metrics = self.compute_common_losses_and_metrics(batch, preds['inv_depths'], poses, progress)

                if 'cam_disp' in preds and 'lidar_disp' in preds:
                    st_losses, st_metrics = self_teaching_loss_handler(batch, preds, poses, progress,
                                                                       self._monoscale_hinted_loss,
                                                                       self._selfteaching_loss)
                    losses += st_losses
                    metrics.update(st_metrics)

            total_loss = sum(losses)

            return { **preds, 'loss': total_loss, 'metrics': metrics}


    def training_step(self, batch, *args):
        """

        Parameters
        ----------
        batch: (Tensor | (Tensor, …) | [Tensor, …])
            The output of your DataLoader. A tensor, tuple or list.

        batch_idx: int
            Integer displaying index of this batch

        optimizer_idx: int
            When using multiple optimizers, this argument will also be present.

        Note: As we use multiple optimizers, training_step() will have an additional optimizer_idx parameter.

        Returns
        -------
        Dict with loss key and optional log or progress bar keys.
        When implementing training_step(), return whatever you need in that step:

            loss -> tensor scalar *****REQUIRED*****

            progress_bar -> Dict for progress bar display. Must have only tensors (no images, etc)

            log -> Dict of metrics to add to logger. Must have only tensors (no images, etc)

            others....
        """

        output = self(batch)

        log_loss = output['loss']
        log_metrics = copy.deepcopy(output['metrics'])

        if self.hparams.logger == WANDB_LOGGER_KEY:
            logs = {
                'train_loss': log_loss,
                'metrics': log_metrics
            }

        elif self.hparams.logger == TENSORBOARD_LOGGER_KEY:
            # in PL 0.8.1 can't log nested metrics
            # so need to flatten and group using slash syntax
            log_metrics = {'train/'+k: v for k,v in log_metrics.items()}
            logs = {
                'train/full_loss': log_loss,
                **log_metrics
            }
        else:
            logs = {'train_loss': log_loss}

        results = {
            'loss': output['loss'],
            'log': logs,
            'progress_bar': {'full_loss': log_loss}
        }

        return results

    def validation_step(self, batch, batch_idx):
        output = self.evaluate_depth(batch)
        # output contains 'metrics' tensor average over the batch and 'inv_depth' the predicted inverse depth maps

        if self.hparams.logger == WANDB_LOGGER_KEY:
            images = wandb_prep_images('val', batch, output, batch_idx, self.hparams.log_images_interval)
        elif self.hparams.logger == TENSORBOARD_LOGGER_KEY:
            images = tensorboard_prep_images('val', batch, output, batch_idx, self.hparams.log_images_interval)
        else:
            images = {}

        return {'images': images, 'metrics': output['metrics']}


    def validation_epoch_end(self, outputs):
        """
        Called at the end of the validation epoch with the outputs of all validation steps.

        Note:
            - The outputs here are strictly for logging or progress bar.
            - If you don’t need to display anything, don’t return anything.
            - If you want to manually set current step, you can specify the ‘step’ key in the ‘log’ dict.

        Further details at:
        https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#pytorch_lightning.core.LightningModule

        Parameters
        ----------
        outputs : (Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]])
            List of outputs you defined in validation_step(),
            or if there are multiple dataloaders, a list containing a list of outputs for each dataloader.

        Returns
        -------
        Dict or OrderedDict. May have the following optional keys:
            `progress_bar` (dict for progress bar display; only tensors)
            `log` (dict of metrics to add to logger; only tensors).

        """
        list_of_metrics = [output['metrics'] for output in outputs]
        avg_metrics_values = average_metrics(list_of_metrics,
                                             prefix='val')

        aggregated_images = {}
        list_of_images_dict = [output['images'] for output in outputs]
        for images_dict in list_of_images_dict:
            aggregated_images.update(images_dict)

        if self.hparams.logger == WANDB_LOGGER_KEY:
            logs = {**aggregated_images, **avg_metrics_values}

        elif self.hparams.logger == TENSORBOARD_LOGGER_KEY:
            for images_title, figure in aggregated_images.items():
                self.logger.experiment.add_figure(images_title, figure, global_step=self.current_epoch)
            logs = avg_metrics_values
        else:
            logs = avg_metrics_values

        results = {
            'val-rmse_log': avg_metrics_values['val/rmse_log'],
            'log': logs,
            'progress_bar': {'rmse_log': avg_metrics_values['val/rmse_log'],
                             'a1': avg_metrics_values['val/a1']
                             }
        }

        return results

    def test_step(self, batch, batch_idx):
        output = self.evaluate_depth(batch)
        # output contains 'metrics' tensor average over the batch and 'inv_depth' the predicted inverse depth maps

        if self.hparams.logger == WANDB_LOGGER_KEY:
            images = wandb_prep_images('test', batch, output, batch_idx, self.hparams.log_images_interval)
        elif self.hparams.logger == TENSORBOARD_LOGGER_KEY:
            images = tensorboard_prep_images('test', batch, output, batch_idx, self.hparams.log_images_interval)
        else:
            images = {}

        return {'images': images, 'metrics': output['metrics']}

    def test_epoch_end(self, outputs):
        list_of_metrics = [output['metrics'] for output in outputs]
        avg_metrics_values = average_metrics(list_of_metrics,
                                             prefix='test')

        aggregated_images = {}
        for dict in [output['images'] for output in outputs]:
            aggregated_images.update(dict)

        if self.hparams.logger == WANDB_LOGGER_KEY:
            logs = {**aggregated_images, **avg_metrics_values}

        elif self.hparams.logger == TENSORBOARD_LOGGER_KEY:
            for images_title, figure in aggregated_images.items():
                self.logger.experiment.add_figure(images_title,figure, global_step=self.current_epoch)
            logs = avg_metrics_values

        else:
            logs = avg_metrics_values

        results = {
            'test-rmse_log': avg_metrics_values['test/rmse_log'],
            'log': logs,
            'progress_bar': {'rmse_log': avg_metrics_values['test/rmse_log'],
                             'a1': avg_metrics_values['test/a1']
                             }
        }


        return results

    def on_train_end(self):
        self.trainer.checkpoint_callback._save_model(filepath=self.trainer.checkpoint_callback.dirpath+'/latest.ckpt')


    def configure_optimizers(self):
        """
        method required by pytorch lightning's module

        Here we use the fact that Every optimizer of pytorch can take as argument a list of dict.
        Each dict defining a separate parameter group, and should contain a `params` key, containing a list of
        parameters belonging to it. Other keys should match the keyword arguments accepted by the optimizers,
        and will be used as optimization options for this group.


        Returns
        -------
            One or multiple optimizers and learning_rate schedulers in any of these options:

                - Single optimizer.
                - List or Tuple - List of optimizers.
                - Two lists - The first list has multiple optimizers, the second a list of LR schedulers.
                - Dictionary, with an ‘optimizer’ key and (optionally) a ‘lr_scheduler’ key.
                - Tuple of dictionaries as described, with an optional ‘frequency’ key.
                - None - Fit will run without any optimizer.

        more details on:
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.html
        at configure_optimizers()
        """

        # REQUIRED
        if self.hparams.optimizer.name == 'Ranger':
            from ranger import Ranger
            optimizer_class = Ranger
        elif self.hparams.optimizer.name == 'RAdam':
            from radam import RAdam
            optimizer_class = RAdam
        else:
            optimizer_class = getattr(torch.optim, self.hparams.optimizer.name)

        params = []
        if self.depth_net is not None:
            params.append({
                'name': 'Depth',
                'params': self.depth_net.parameters(),
                **self.hparams.optimizer.depth_net_options
            })
            terminal_logger.info("DepthNet's optimizer configured.")

        if self.pose_net is not None:
            params.append({
                'name': 'Pose',
                'params': self.pose_net.parameters(),
                **self.hparams.optimizer.pose_net_options
            })
            terminal_logger.info("PoseNet's optimizer configured.")

        # Create optimizer with parameters
        optimizer = optimizer_class(params)

        # Load and initialize schedulers
        if self.hparams.scheduler.name == 'FlatCosAnnealScheduler':
            from schedulers.flat_cos_anneal_scheduler import FlatCosAnnealScheduler
            step_factor = self.hparams.dataloaders.train.batch_size * self.hparams.trainer.accumulate_grad_batches

            scheduler = {
                'scheduler': FlatCosAnnealScheduler(optimizer, step_factor, len(self.train_dataset),
                                                    self.hparams.trainer.max_epochs,
                                                    **self.hparams.scheduler.options),
                'name': 'FlatCosAnnealScheduler',
                'interval': 'step',  # so that scheduler.step() is done at batch-level instead of epoch
                'frequency': 1
            }

        else:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.hparams.scheduler.name)
            # assumes the schedulers used from torch.optim are epoch-based
            scheduler = {
                'scheduler': scheduler_class(optimizer, **self.hparams.scheduler.options),
                'name': self.hparams.scheduler.name,
                'interval': 'epoch',
                'frequency': 1
            }


        terminal_logger.info("Optimizers and Schedulers configured.")

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def custom_collate(self, batch):
        """
        Override `default_collate` https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader

        Reference:
        def default_collate(batch) at https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
        https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
        https://github.com/pytorch/pytorch/issues/1512

        We need our own collate function that wraps things up (imge, mask, label).

        In this setup,  batch is a list of tuples (the result of calling: img, mask, label = Dataset[i].
        The output of this function is four elements:
            . data: a pytorch tensor of size (batch_size, c, h, w) of float32 . Each sample is a tensor of shape (c, h_,
            w_) that represents a cropped patch from an image (or the entire image) where: c is the depth of the patches (
            since they are RGB, so c=3),  h is the height of the patch, and w_ is the its width.
            . mask: a list of pytorch tensors of size (batch_size, 1, h, w) full of 1 and 0. The mask of the ENTIRE image (no
            cropping is performed). Images does not have the same size, and the same thing goes for the masks. Therefore,
            we can't put the masks in one tensor.
            . target: a vector (pytorch tensor) of length batch_size of type torch.LongTensor containing the image-level
            labels.
        :param batch: list of tuples (img, mask, label)
        :return: 3 elements: tensor data, list of tensors of masks, tensor of labels.
        """

        continuous_conv_collate = batch[0].get('nn_pixel_idxs',None) is not None

        if not continuous_conv_collate:
            return default_collate(batch)

        nb_scales = len(batch[0]['nn_pixel_idxs'])

        # changing order of access, in: batch[n]['nn_pixel_idxs][s] -> out: batch['nn_pixel_idxs][s][n]
        nn_pixel_idxs = [[] for _ in range(nb_scales)]
        pixel_idxs = [[] for _ in range(nb_scales)]
        nn_diff_pts_3d = [[] for _ in range(nb_scales)]

        for elem in batch:
            tmp_nn_pixel_idxs = elem.pop('nn_pixel_idxs')
            tmp_pixel_idxs = elem.pop('pixel_idxs')
            tmp_diff_pts_3d = elem.pop('nn_diff_pts_3d')

            for s in range(nb_scales):
                nn_pixel_idxs[s].append(tmp_nn_pixel_idxs[s])
                pixel_idxs[s].append(tmp_pixel_idxs[s])
                nn_diff_pts_3d[s].append(tmp_diff_pts_3d[s])

        # collating by scale
        nn_pixel_idxs = [default_collate(scale) for scale in nn_pixel_idxs]
        pixel_idxs = [default_collate(scale) for scale in pixel_idxs]
        nn_diff_pts_3d = [default_collate(scale) for scale in nn_diff_pts_3d]

        out = default_collate(batch)

        out['nn_pixel_idxs'] = nn_pixel_idxs
        out['pixel_idxs'] = pixel_idxs
        out['nn_diff_pts_3d'] = nn_diff_pts_3d

        return out


    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.dataloaders.train.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=16,
                          collate_fn=self.custom_collate
                          )


    def val_dataloader(self):
        # REQUIRED
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.dataloaders.val.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=16,
                          collate_fn=self.custom_collate
                          )
    def test_dataloader(self):
        # REQUIRED
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.dataloaders.test.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=16,
                          collate_fn=self.custom_collate
                          )