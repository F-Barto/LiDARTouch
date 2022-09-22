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
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import hydra

from lidartouch.utils.wandb_logging import prepare_images_to_log as wandb_prep_images
from lidartouch.utils.tensorboard_logging import prepare_images_to_log as tensorboard_prep_images
from lidartouch.utils.image import interpolate_scales, flip_lr
from lidartouch.utils.misc import make_list
from lidartouch.utils.depth import inv2depth, compute_depth_metrics
from lidartouch.utils.common_logging import average_metrics
from lidartouch.utils.identifiers import INV_DEPTHS, GT_DEPTH, SPARSE_DEPTH, IMAGE, TARGET_VIEW, depthnet_to_batch_keys


TENSORBOARD_LOGGER_KEY = 'tensorboard'
WANDB_LOGGER_KEY = 'wandb'


EXAMPLES_TO_LOG = [
    '2011_09_26_0011_c2_0000000077',
    '2011_09_26_0039_c2_0000000173',
    '2011_09_26_0051_c2_0000000381'
]


class BaseDepthEstimationModel(pl.LightningModule):

    def __init__(self,
                 depth_net: dict,
                 depth_optimizer_conf: dict,
                 depth_scheduler_conf: dict,
                 smooth_loss: dict,
                 metrics_hparams: dict,
                 depth_scheduler_interval: str = 'epoch',
                 train_flip_input: bool = True,
                 logger: str = 'tensorboard',
                 train_log_images_interval: int = 0,
                 val_log_images_interval: int = 0,
                 upsample_depth_maps: bool = True,
                 *args, **kwargs):
        """
        Base class from which all other learning systems for depth prediction must inherit

        Args:
            depth_optimizer_conf: Dict of the form {"class_path":...,"init_args":...} used to instantiate the depth
                network's optimizer. The dict is typically populated by the CLI.
            depth_scheduler_conf: Dict of the form {"class_path":...,"init_args":...} used to instantiate the
                depth network's lr scheduling. The dict is typically populated by the CLI.
            train_flip_input: whether to flip input randomly at train time or not
            logger: name of logging system to use, current choices are ['tensorboard', 'wandb']
            train_log_images_interval: number of steps between two consecutive logging of images during train phase
            val_log_images_interval: number of steps between two consecutive logging of images during val phase
            upsample_depth_maps: whether to upsample the multiscale depth predictions to the biggest scale (like MD2)
            metrics_hparams: dict of options for depth metrics of predicted depth maps
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.train_flip_input = train_flip_input
        self.logger_name = logger
        self.train_log_images_interval = train_log_images_interval
        self.val_log_images_interval = val_log_images_interval
        self.upsample_depth_maps = upsample_depth_maps
        self.metrics_hparams = metrics_hparams

        self.depth_optimizer_conf = depth_optimizer_conf
        self.depth_scheduler_conf = depth_scheduler_conf
        self.depth_scheduler_interval = depth_scheduler_interval

        self.depth_net = hydra.utils.instantiate(depth_net)
        self.smoothness_loss = hydra.utils.instantiate(smooth_loss)

        ################### HPARAMS Validation #####################
        assert self.logger_name in [TENSORBOARD_LOGGER_KEY, WANDB_LOGGER_KEY, None], \
            f"Logger should either be {TENSORBOARD_LOGGER_KEY}, {WANDB_LOGGER_KEY} or None"

    def configure_optimizers(self):

        if not self.depth_optimizer_conf:
            return None

        optimizer = hydra.utils.instantiate(self.depth_optimizer_conf, params=self.depth_net.parameters())

        if not self.depth_scheduler_conf:
            return optimizer

        scheduler = hydra.utils.instantiate(self.depth_scheduler_conf, optimizer=optimizer)

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": self.depth_scheduler_interval,
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            # "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            # "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": 'lr_depthnet',
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def compute_inv_depths(self, batch):
        """Computes inverse depth maps from single images.

        1- If in training loop and argument `train_flip_input` is True, randomly horizontaly flip inputs in camera plane
        2- Map kwargs name to `depth_net` signature:
            e.g. key in batch is 'projected_lidar' but `depth_net` argument's name is 'sparse_depth'
        3- Prediction are made with flipped inputs, so we need to unflip predictions.
        4- Following MonoDepth2, we upsample depth prediction at each scale to biggest resolution.
        5- Return predictions in `outputs` dict
        """

        flip = random.random() < 0.5 if self.training and self.train_flip_input else False

        required_inputs = self.depth_net.required_inputs
        # e.g. key in batch is 'projected_lidar' but `depth_net` arg name is 'sparse_depth'
        inputs = {input_key: batch[depthnet_to_batch_keys[input_key]] for input_key in required_inputs}

        if flip:
            keys_to_flip = [IMAGE, SPARSE_DEPTH]
            # only keep keys of inputs that are both required as input and amenable to be flipped
            keys_to_flip = [input_key for input_key in required_inputs if input_key in keys_to_flip]
            for input_key in keys_to_flip:
                inputs[input_key] = flip_lr(inputs[input_key])

        output = self.depth_net(**inputs)


        ################# Post-process outputs #################

        # We need to unflip output in case of photometric error against original (unflipped) input
        keys_to_unflip = [INV_DEPTHS]
        multiscale_keys = [INV_DEPTHS] # following MonoDepth2, multi scale depth pred. are upscaled to biggest scale

        for key in keys_to_unflip:
            if key not in output:
                continue

            output[key] = make_list(output[key])

            if flip:
                output[key] = [flip_lr(o) for o in output[key]]

            # interpolate preds from all scales to biggest scale
            if key in multiscale_keys and self.upsample_depth_maps:
                output[key] = interpolate_scales(output[key], mode='nearest')

        return output

    def compute_losses_and_metrics(self, batch, predictions):
        smoothness_loss, smoothness_metrics = self.smoothness_loss(predictions[INV_DEPTHS],
                                                                   batch[TARGET_VIEW+'_original'])

        return smoothness_loss, smoothness_metrics


    def training_step(self, batch, batch_idx, *args):
        output = self(batch)

        out_dict = {}

        self.log_dict(output['metrics'])

        log_loss = output['loss'].item()
        self.log('train_loss', log_loss, prog_bar=True)

        checks = [f in EXAMPLES_TO_LOG for f in batch['filename']]
        if any(checks):
            images = self.prep_images(batch, output, batch_idx, 'train', 1, i=checks.index(True))#self.train_log_images_interval)
            out_dict['images'] = images
        else:
            out_dict['images'] = {}

        out_dict['loss'] = output['loss']

        return out_dict


    def training_epoch_end(self, outputs):

        print('/'*60)
        print(len(outputs))
        print('/' * 60)

        #if len(outputs) == 2:
        #    outputs = outputs[0]

        print(outputs)

        self.log_images(outputs)



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

        inv_depth = output[INV_DEPTHS][0]
        depth = inv2depth(inv_depth)

        # Calculate predicted metrics
        metrics = compute_depth_metrics(gt=batch[GT_DEPTH], pred=depth, **self.metrics_hparams)
        # Return metrics and extra information

        result = {
            'metrics': metrics,
            'inv_depth': inv_depth,
            'depth': depth,
        }

        return result


    def validation_step(self, batch, batch_idx):
        output = self.evaluate_depth(batch)
        # output contains 'metrics' tensor average over the batch and 'inv_depth' the predicted inverse depth maps

        images = self.prep_images(batch, output, batch_idx, 'val', self.val_log_images_interval)

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

        avg_metrics_values = average_metrics(list_of_metrics, prefix='val_')

        self.log_images(outputs)

        # to logger (e.g, tensorboard)
        self.log_dict(avg_metrics_values)

        # log in progress bar and also log value for ``val_loss` key used as criterion for checkpoint and lr scheduling
        self.log('val_loss', avg_metrics_values['val_rmse_log'], prog_bar=True, logger=False)
        self.log('a1', avg_metrics_values['val_a1'], prog_bar=True, logger=False)


    def test_step(self, batch, batch_idx):
        output = self.evaluate_depth(batch)
        # output contains 'metrics' tensor average over the batch and 'inv_depth' the predicted inverse depth maps

        images = self.prep_images(batch, output, batch_idx, 'test', self.val_log_images_interval)

        return {'images': images, 'metrics': output['metrics']}


    def test_epoch_end(self, outputs):
        list_of_metrics = [output['metrics'] for output in outputs]

        avg_metrics_values = average_metrics(list_of_metrics, prefix='test_')

        self.log_images(outputs)

        # to logger (e.g, tensorboard)
        self.log_dict(avg_metrics_values)


    def prep_images(self, batch, output, batch_idx, phase, log_images_interval, i=0):

        if self.logger_name == WANDB_LOGGER_KEY:
            images = wandb_prep_images(phase, batch, output, batch_idx, log_images_interval)
        elif self.logger_name == TENSORBOARD_LOGGER_KEY:
            images = tensorboard_prep_images(phase, batch, output, batch_idx, log_images_interval, i=i)
        elif self.logger_name is None:
            images = {}
        else:
            raise NotImplementedError

        return images

    def log_images(self, outputs):

        aggregated_images = {}

        list_of_images_dict = [output['images'] for output in outputs if output['images'] != {}]
        for images_dict in list_of_images_dict:
            aggregated_images.update(images_dict)

        if self.logger_name == WANDB_LOGGER_KEY:
            self.logger.experiment.log({**aggregated_images})
        elif self.logger_name == TENSORBOARD_LOGGER_KEY:
            for images_title, figure in aggregated_images.items():
                self.logger.experiment.add_figure(images_title, figure, global_step=self.current_epoch)

            plt.close('all')
        elif self.logger_name is None:
            pass
        else:
            raise NotImplementedError

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_class_path'] = self.__module__ + '.' + self.__class__.__qualname__

    #def on_load_checkpoint(self, checkpoint):
    #    print(checkpoint)

