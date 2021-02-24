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

from pytorch_lightning import _logger as terminal_logger
from pytorch_lightning.core.decorators import auto_move_data

from models.model_utils import select_depth_net, select_pose_net

from losses.handlers.multiview_loss_handler import MultiViewLossHandler
from losses.handlers.handler_base import LossHandler

from utils.pose import Pose
from utils.image import interpolate_scales, flip_lr
from utils.depth import inv2depth, compute_depth_metrics

from utils.common_logging import average_metrics
from utils.loading import load_tri_network
from utils.misc import make_list

from models.model_base import BaseModel

IMPLEMENTED_ROTATION_MODES = ['euler']

class MultiViewModel(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)

        ################### Networks Definition #####################

        # Depth Net
        self.depth_net = select_depth_net(self.hparams.network.depth_net.name, self.hparams.network.depth_net.options,
                                          self.train_dataset.load_sparse_depth)

        # Pose Net
        if hasattr(self.hparams.datasets.train, 'use_pnp') and self.hparams.datasets.train.use_pnp:
            self.pose_net = None
        else:
            self.pose_net = select_pose_net(self.hparams.network.pose_net.name, hparams.network.pose_net.options)

        ################### Checkpoint loading Definition #####################

        tri_checkpoint_path =  self.hparams.network.get('tri_checkpoint_path', None)
        if tri_checkpoint_path is not None:
            load_tri_network(self, tri_checkpoint_path)

        ################### Losses Definition #####################

        self.multi_view_loss_handler = MultiViewLossHandler(self.hparams.losses)
        other_losses_handler = LossHandler(self.hparams.losses).parse_all_losses()

        self.velocity_loss_handler = None
        if 'velocity' in other_losses_handler:
            self.velocity_loss_handler = other_losses_handler['velocity']

        self.regression_loss_handler = None
        if 'regression' in other_losses_handler:
            self.regression_loss_handler = other_losses_handler['regression']

        ################### HPARAMS Validation #####################

        assert hparams.rotation_mode in IMPLEMENTED_ROTATION_MODES, \
            f"Option `rotation_mode` should be in {IMPLEMENTED_ROTATION_MODES}"
        self.rotation_mode = hparams.rotation_mode


    def compute_common_losses_and_metrics(self, batch, disp_preds, poses_preds, progress=0.0, metrics_prefix=''):
        losses = []
        metrics = {}

        mv_outputs = self.multi_view_loss_handler(
            batch['target_view_original'],
            batch['source_views_original'],
            disp_preds,
            batch['intrinsics'],
            poses_preds,
            batch['sparse_projected_lidar_original'],
            progress=progress
        )

        losses.append(mv_outputs['loss'])
        metrics.update({metrics_prefix + k: v for k, v in mv_outputs['metrics'].items()})

        if self.regression_loss_handler is not None:
            # Calculate and weight supervised loss
            sup_output = self.regression_loss_handler(disp_preds, batch['sparse_projected_lidar_original'],
                                                      progress=progress)
            losses.append(sup_output['loss'])
            metrics.update({metrics_prefix + k: v for k, v in sup_output['metrics'].items()})

        if self.velocity_loss_handler is not None:
            translation_magnitudes = batch.get('translation_magnitudes', None)
            velocity_output = self.velocity_loss_handler(poses_preds, translation_magnitudes)
            losses.append(velocity_output['loss'])
            metrics.update({metrics_prefix + k: v for k, v in velocity_output['metrics'].items()})

        return losses, metrics


    def compute_inv_depths(self, image, sparse_depth=None, intrinsics=None):
        """Computes inverse depth maps from single images"""

        flip = random.random() < 0.5 if self.training else False

        if flip:
            image = flip_lr(image) if self.depth_net.require_image_input else image
            sparse_depth = flip_lr(sparse_depth) if self.depth_net.require_lidar_input else sparse_depth

        if self.depth_net.require_lidar_input and self.depth_net.require_image_input \
                and hasattr(self.depth_net, 'require_intrinsics'):
            output = self.depth_net(image, sparse_depth, intrinsics)
        elif self.depth_net.require_lidar_input and self.depth_net.require_image_input:
            output = self.depth_net(image, sparse_depth)
        elif self.depth_net.require_lidar_input and not self.depth_net.require_image_input:
            output = self.depth_net(sparse_depth)
        elif not self.depth_net.require_lidar_input and self.depth_net.require_image_input:
            output = self.depth_net(image)
        else:
            raise NotImplementedError

        # Handle outputs

        keys = ['inv_depths', 'disp', 'coarse_disp', 'cam_disp', 'lidar_disp']

        if 'uncertainties' in output:
            keys.append('uncertainties')

        for key in keys:
            if key not in output:
                continue

            if flip:
                output[key] = [flip_lr(o) for o in make_list(output[key])]
            else:
                output[key] = make_list(output[key])

            # interpolate preds from all scales to biggest scale
            if key in ['inv_depths', 'uncertainties'] and self.hparams.upsample_depth_maps:
                output[key] = interpolate_scales(output[key], mode='nearest')
                # 'cam_disp', 'disp', 'lidar_disp' are one scales preds so we don't have to interpolate

            return output


    def compute_poses(self, target_view, source_views):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.pose_net(target_view, source_views)
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode) for i in range(pose_vec.shape[1])]


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

        output_key = 'inv_depths'

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

        sparse_lidar = batch.get('sparse_projected_lidar', None)
        output = self.compute_inv_depths(batch['target_view'], sparse_depth=sparse_lidar, intrinsics=batch['intrinsics'])

        if 'poses_pnp' in batch:
            pose_vec = batch['poses_pnp'].float()
            poses = [Pose.from_vec(pose_vec[:, i], self.rotation_mode) for i in range(pose_vec.shape[1])]
        elif 'source_views' in batch and self.pose_net is not None:
            poses = self.compute_poses(batch['target_view'], batch['source_views'])
        else:
            poses = None

        preds = {
            'poses': poses,
        }
        preds.update(output)

        if not self.training:
            return preds
        else:
            progress = self.current_epoch / self.hparams.trainer.max_epochs

            losses, metrics = self.compute_common_losses_and_metrics(batch, preds['inv_depths'], poses, progress)

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

        log_losses = output['loss']
        log_metrics = copy.deepcopy(output['metrics'])

        logs = self.log_train(log_losses, log_metrics)

        results = {
            'loss': output['loss'],
            'log': logs,
            'progress_bar': {'full_loss': log_losses}
        }

        return results


    def validation_step(self, batch, batch_idx):
        output = self.evaluate_depth(batch)
        # output contains 'metrics' tensor average over the batch and 'inv_depth' the predicted inverse depth maps

        images = self.log_images(batch, output, batch_idx, 'val')

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

        logs = self.log_val_test(outputs, avg_metrics_values)

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

        images = self.log_images(batch, output, batch_idx, 'test')

        return {'images': images, 'metrics': output['metrics']}


    def test_epoch_end(self, outputs):
        list_of_metrics = [output['metrics'] for output in outputs]
        avg_metrics_values = average_metrics(list_of_metrics,
                                             prefix='test')

        logs = self.log_val_test(outputs, avg_metrics_values)

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

        scheduler = self.configure_scheduler(optimizer)

        terminal_logger.info("Optimizers and Schedulers configured.")

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}




