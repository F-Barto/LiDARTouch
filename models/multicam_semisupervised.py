'''
This module houses:

Model definition (init)
Computations (forward)
What happens inside the training loop (training_step)
What happens inside the validation loop (validation_step)
What optimizer(s) to use (configure_optimizers)
What data to use (train_dataloader, val_dataloader, test_dataloader)

'''

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import _logger as terminal_logger


from argoverse.utils.camera_stats import (
    CAMERA_LIST,
    RING_CAMERA_LIST,
    STEREO_CAMERA_LIST,
)


from networks.legacy.packnet.packnet import PackNet01
from networks.legacy.packnet.posenet import PoseNet
from networks.legacy.monodepth2.depth_rest_net import DepthResNet
from networks.legacy.monodepth2.pose_res_net import PoseResNet

from losses.legacy.multicam_photometric_loss import MultiCamPhotometricLoss
from losses.elements.supervised_loss import ReprojectedLoss

from dataloaders.multicam_argoverse import SequentialArgoverseLoader
from dataloaders.transforms import train_transforms, val_transforms, test_transforms

from utils.pose import Pose
from utils.image import interpolate_scales
from utils.depth import inv2depth, compute_depth_metrics
from utils.wandb_logging import prepare_images_to_log, average_metrics
from utils.config_utils import YParams
from utils.types import is_list
from utils.loading import load_tri_network




IMPLEMENTED_ROTATION_MODES = ['euler']

def validate_camera_option(cameras: str):
    """
    Just checks the list of cameras names is valid and return a list without duplicates
    Assumes `cameras` is a comma separated list of camera names
    """
    camera_set = set()

    cameras = cameras.split(',')
    for camera in cameras:
        if camera == "ring":
            camera_set.update(set(RING_CAMERA_LIST)) # inplace union
        elif camera == "stereo":
            camera_set.update(set(STEREO_CAMERA_LIST))
        elif camera in CAMERA_LIST:
            camera_set.add(camera)
        else:
            raise ValueError(f"Camera of name {camera} is not valid. Cameras available: {CAMERA_LIST}")

    return list(camera_set)

def prepare_data(datasets_config, camera_list=None, input_channels=3):
    terminal_logger.info("Preparing Datasets...")

    train_dataset = SequentialArgoverseLoader(**datasets_config.train, camera_list=camera_list,
                                              data_transform=train_transforms, input_channels=input_channels)

    val_dataset = SequentialArgoverseLoader(**datasets_config.val, camera_list=camera_list,
                                            data_transform=val_transforms, input_channels=input_channels)

    test_dataset = SequentialArgoverseLoader(**datasets_config.test, camera_list=camera_list,
                                             data_transform=test_transforms, input_channels=input_channels)

    return train_dataset, val_dataset, test_dataset

class MonoSemiSupDepth_Packnet(pl.LightningModule):
    def __init__(self, _hparams,):
        super().__init__()

        # since pytorch-lightning >= 0.7.6 the Checkpoint callback only accepts dict type as self.hparams's type
        # self.hparams is a dict used for checkpoint
        # self._hparams is the YParams object used internally
        if _hparams.__class__.__name__ == "dict":
            self.hparams = _hparams
            self._hparams = YParams.fromDict(_hparams)
        elif _hparams.__class__.__name__ == "YParams":
            self.hparams = _hparams.toDict()
            self._hparams = _hparams
        else:
            raise ValueError(
                'The acceptable hparams type is dict or YParams,',
                f' not {_hparams.__class__.__name__}'
            )

        self.camera_list = validate_camera_option(_hparams.camera_list)

        train_dataset, val_dataset, test_dataset = prepare_data(_hparams.datasets, self.camera_list,
                                                                input_channels=_hparams.input_channels)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.input_channels = _hparams.input_channels


        assert _hparams.rotation_mode in IMPLEMENTED_ROTATION_MODES, \
            f"Option `rotation_mode` should be in {IMPLEMENTED_ROTATION_MODES}"
        self.rotation_mode = _hparams.rotation_mode


        ################### Model Definition #####################

        # Depth Net
        if self._hparams.model.depth_net.name == 'packnet':
            self.depth_net = PackNet01(**_hparams.model.depth_net.options, input_channels=self.input_channels)
        elif self._hparams.model.depth_net.name == 'monodepth':
            self.depth_net = DepthResNet(**_hparams.model.depth_net.options)
        else:
            terminal_logger.error(f"Depth net {self._hparams.model.depth_net.name} not implemented")

        # Pose Net
        if self._hparams.model.pose_net.name == 'packnet':
            self.pose_net = PoseNet(**_hparams.model.pose_net.options, input_channels=self.input_channels)
        elif self._hparams.model.pose_net.name == 'monodepth':
            self.pose_net = PoseResNet(**_hparams.model.pose_net.options)
        else:
            terminal_logger.error(f"Pose net {self._hparams.model.pose_net.name} not implemented")

        tri_checkpoint_path =  self._hparams.model.get('tri_checkpoint_path', None)
        if tri_checkpoint_path is not None:
            load_tri_network(self, tri_checkpoint_path)

        # Photometric loss used as main supervisory signal
        self.photometric_losses = {camera_name : MultiCamPhotometricLoss(**_hparams.losses.MultiCamPhotometricLoss)
                                   for camera_name in self.camera_list}


        if self._hparams.semi_supervised == True:
            self.reprojected_losses = {camera_name : ReprojectedLoss for camera_name in self.camera_list}
            self.supervised_loss_weight = _hparams.losses.supervised_loss_weight

    def compute_inv_depths(self, image):
        """Computes inverse depth maps from single images"""

        inv_depths = self.depth_net(image)
        inv_depths = inv_depths if is_list(inv_depths) else [inv_depths]

        # already done in loss computation
        if self._hparams.upsample_depth_maps:
            inv_depths = interpolate_scales(inv_depths, mode='nearest')

        # Return inverse depth maps
        return inv_depths

    def compute_poses(self, target_view, source_views):
        """Compute poses from image and a sequence of context images"""
        pose_vec = self.pose_net(target_view, source_views)
        return [Pose.from_vec(pose_vec[:, i], self.rotation_mode) for i in range(pose_vec.shape[1])]

    def self_supervised_loss(self, camera_name, image, ref_images, inv_depths, poses, intrinsics, extrinsics, progress):
        """
        Calculates the self-supervised photometric loss.
        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        ref_images : list of torch.Tensor [B,3,H,W]
            Reference images from context
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        poses : list of Pose
            List containing predicted poses between original and context images
        intrinsics : torch.Tensor [B,3,3]
            Camera intrinsics
        progress :
            Training progress percentage
        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self.photometric_losses[camera_name](image, ref_images, inv_depths, intrinsics,
                                                    extrinsics, poses, progress=progress)

    def supervised_loss(self, camera_name, inv_depths, projected_lidar, intrinsics, poses, progress):
        return self.reprojected_losses[camera_name](inv_depths, projected_lidar, intrinsics, poses, progress=progress)

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

        total_metrics = {}
        output_by_cam = {}

        for i, camera_name in enumerate(self.camera_list):
            output_by_cam[camera_name] = {}

            # Get predicted depth
            inv_depth = self(batch)[camera_name]['inv_depths'][0]
            depth = inv2depth(inv_depth)

            # store predictions for viz purpose
            output_by_cam[camera_name]['inv_depth'] = inv_depth
            output_by_cam[camera_name]['depth']  = depth

            # Calculate predicted metrics, store by camera & averaged over all cameras
            metric_prefix = f"{camera_name}-"
            metrics = compute_depth_metrics(gt=batch[camera_name]['projected_lidar'], pred=depth,
                                            prefix=metric_prefix, **self._hparams.metrics)

            total_metrics.update(metrics)

            for key,val in metrics.items():
                key = key[len(metric_prefix):]
                total_metrics[key] = total_metrics.get(key, torch.zeros_like(val)) + val
                if i == len(self.camera_list) - 1:
                    total_metrics[key] /= len(self.camera_list) # average over all cameras at the end

        # Return metrics averaged by cam and extra information
        return {
            'metrics': total_metrics,
            **output_by_cam
        }

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

        output = {}
        losses_and_metrics = {}

        for i, camera_name in enumerate(self.camera_list):

            inv_depths = self.compute_inv_depths(batch[camera_name]['target_view'])

            poses = None
            if 'source_views' in batch[camera_name] and self.pose_net is not None:
                poses = self.compute_poses(batch[camera_name]['target_view'], batch[camera_name]['source_views'])

            output[camera_name] = {
                'inv_depths': inv_depths,
                'poses': poses,
            }

            if self.training:
                progress = self.current_epoch / self.trainer.max_epochs
                self_sup_output = self.self_supervised_loss(
                    camera_name,
                    batch[camera_name]['target_view_original'],
                    batch[camera_name]['source_views_original'],
                    output[camera_name]['inv_depths'],
                    output[camera_name]['poses'],
                    batch[camera_name]['intrinsics'],
                    batch[camera_name]['extrinsics'],
                    progress=progress)

                if self._hparams.semi_supervised == False:
                    losses_and_metrics[camera_name] = self_sup_output
                else:
                    # Calculate and weight supervised loss
                    sup_output = self.supervised_loss(output[camera_name]['inv_depths'],
                                                      batch[camera_name]['projected_lidar'],
                                                      batch[camera_name]['intrinsics'],
                                                      output[camera_name]['poses'], progress=progress)

                    loss = self_sup_output['loss'] + self.supervised_loss_weight * sup_output['loss']

                    # merging both output dicts
                    metrics = {**self_sup_output['metrics'], **sup_output['metrics']}
                    losses_and_metrics[camera_name] = {'loss': loss, 'metrics': metrics}


        if self.training:

            nb_cameras = len(self.camera_list)

            if self._hparams.losses.cam_reduction == 'avg':
                loss = sum([losses_and_metrics[camera_name]['loss'] for camera_name in self.camera_list]) / nb_cameras
            elif self._hparams.losses.cam_reduction == 'min':
                loss = min([losses_and_metrics[camera_name]['loss'] for camera_name in self.camera_list])
            else:
                raise ValueError("hparam: losses.cam_reduction should be either 'avg' or 'min'.")

            total_metrics = {}
            for i, camera_name in enumerate(self.camera_list):
                for k,v in losses_and_metrics[camera_name]['metrics'].items():
                    total_metrics[k] = total_metrics.get(k, torch.zeros_like(v)) + v
                    total_metrics[f'{camera_name}_{k}'] = v
                    if i == nb_cameras - 1:
                        total_metrics[k] /= nb_cameras # average over all cameras at the end

            return { **output, 'loss': loss, 'metrics': total_metrics}
        else:
            return output


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

        wandb_logs = {
            'train_loss': output['loss'],
            'metrics': output['metrics']
        }

        results = {
            'loss': output['loss'],
            'log': wandb_logs,
            'progress_bar': {'train_loss': output['loss'], **output['metrics']}
        }

        return results

    def validation_step(self, batch, batch_idx):
        output = self.evaluate_depth(batch)
        # output contains 'metrics' tensor average over the batch and 'inv_depth' the predicted inverse depth maps

        images = {}
        for camera_name in self.camera_list:
            images.update(prepare_images_to_log('val', batch[camera_name], output[camera_name], batch_idx,
                                                self._hparams.log_images_interval))

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

        wandb_logs = {**aggregated_images ,**avg_metrics_values}

        results = {
            'val-abs_rel': avg_metrics_values['val-abs_rel'],
            'log': wandb_logs,
            'progress_bar': wandb_logs
        }

        return results

    def test_step(self, batch, batch_idx):
        output = self.evaluate_depth(batch)
        # output contains 'metrics' tensor average over the batch and 'inv_depth' the predicted inverse depth maps

        images = {}
        for camera_name in self.camera_list:
            images.update(prepare_images_to_log('test', batch[camera_name], output[camera_name], batch_idx,
                                                self._hparams.log_images_interval))

        return {'images': images, 'metrics': output['metrics']}

    def test_epoch_end(self, outputs):
        list_of_metrics = [output['metrics'] for output in outputs]
        avg_metrics_values = average_metrics(list_of_metrics,
                                             prefix='test')

        aggregated_images = {}
        for dict in [output['images'] for output in outputs]:
            aggregated_images.update(dict)

        wandb_logs = {**aggregated_images, **avg_metrics_values}

        results = {
            'test-abs_rel': avg_metrics_values['test-abs_rel'],
            'log': wandb_logs,
            'progress_bar': wandb_logs
        }
        return results


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
        if self._hparams.optimizer.name == 'Ranger':
            from ranger import Ranger
            optimizer_class = Ranger
        elif self._hparams.optimizer.name == 'RAdam':
            from radam import RAdam
            optimizer_class = RAdam
        else:
            optimizer_class = getattr(torch.optim, self._hparams.optimizer.name)

        params = []
        if self.depth_net is not None:
            params.append({
                'name': 'Depth',
                'params': self.depth_net.parameters(),
                **self._hparams.optimizer.depth_net_options
            })
            terminal_logger.info("DepthNet's optimizer configured.")

        if self.pose_net is not None:
            params.append({
                'name': 'Pose',
                'params': self.pose_net.parameters(),
                **self._hparams.optimizer.pose_net_options
            })
            terminal_logger.info("PoseNet's optimizer configured.")

        # Create optimizer with parameters
        optimizer = optimizer_class(params)

        # Load and initialize schedulers
        if self._hparams.scheduler.name == 'FlatCosAnnealScheduler':
            from schedulers.flat_cos_anneal_scheduler import FlatCosAnnealScheduler
            steps_per_epoch = len(self.train_dataset) / self._hparams.dataloaders.train.batch_size
            scheduler = {
                'scheduler': FlatCosAnnealScheduler(optimizer, steps_per_epoch, self._hparams.trainer.max_epochs,
                                                    **self._hparams.scheduler.options),
                'name': 'FlatCosAnnealScheduler',
                'interval': 'step',  # so that scheduler.step() is done at batch-level instead of epoch
                'frequency': 1
            }

        else:
            scheduler_class = getattr(torch.optim.lr_scheduler, self._hparams.scheduler.name)
            # assumes the schedulers used from torch.optim are epoch-based
            scheduler = {
                'scheduler': scheduler_class(optimizer, **self._hparams.scheduler.options),
                'name': self._hparams.scheduler.name,
                'interval': 'epoch',
                'frequency': 1
            }


        terminal_logger.info("Optimizers and Schedulers configured.")

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset,
                          batch_size=self._hparams.dataloaders.train.batch_size,
                          shuffle=True,
                          pin_memory=False,
                          num_workers=1,
                          )


    def val_dataloader(self):
        # REQUIRED
        return DataLoader(self.val_dataset,
                          batch_size=self._hparams.dataloaders.val.batch_size,
                          shuffle=False,
                          pin_memory=False,
                          num_workers=1,
                          )
    def test_dataloader(self):
        # REQUIRED
        return DataLoader(self.test_dataset,
                          batch_size=self._hparams.dataloaders.test.batch_size,
                          shuffle=False,
                          pin_memory=False,
                          num_workers=1,
                          )