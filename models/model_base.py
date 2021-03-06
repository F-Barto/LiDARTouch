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

from utils.wandb_logging import prepare_images_to_log as wandb_prep_images
from utils.tensorboard_logging import prepare_images_to_log as tensorboard_prep_images

from dataloaders.kitti import SequentialKittiLoader
#from dataloaders.randcam_argoverse import RandCamSequentialArgoverseLoader
from dataloaders.transforms import train_transforms, val_transforms, test_transforms

TENSORBOARD_LOGGER_KEY = 'tensorboard'
WANDB_LOGGER_KEY = 'wandb'

def prepare_data(datasets_config, input_channels=3):
    terminal_logger.info("Preparing Datasets...")

    if datasets_config.dataset_name == 'rand_cam_argoverse':
        # dataset_cls = RandCamSequentialArgoverseLoader
        pass
    elif datasets_config.dataset_name == 'kitti':
        dataset_cls = SequentialKittiLoader
    else:
        raise ValueError(f'Dataset of class {datasets_config.dataset_name} is not implemented')

    train_dataset = dataset_cls(**datasets_config.train,
                                data_transform=train_transforms,
                                input_channels=input_channels)
    print('len train_dataset: ', len(train_dataset))

    val_dataset = dataset_cls(**datasets_config.val,
                              data_transform=val_transforms,
                              input_channels=input_channels)
    print('len val_dataset: ', len(val_dataset))

    test_dataset = dataset_cls(**datasets_config.test,
                               data_transform=test_transforms,
                               input_channels=input_channels)
    print('len test_dataset: ', len(test_dataset))

    return train_dataset, val_dataset, test_dataset

class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()


        self.hparams = hparams

        train_dataset, val_dataset, test_dataset = prepare_data(hparams.datasets, hparams.input_channels)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        ################### HPARAMS Validation #####################

        assert self.hparams.logger in [TENSORBOARD_LOGGER_KEY, WANDB_LOGGER_KEY], \
            f"Logger should either be {TENSORBOARD_LOGGER_KEY} or {WANDB_LOGGER_KEY}"

    def configure_scheduler(self, optimizer):
        """
        helper functions to use insides pytorch lightling's method `configure_optimizers`

        :param optimizer:
        :return:
        """

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

        return scheduler

    def log_train(self, log_losses, log_metrics):
        if self.hparams.logger == WANDB_LOGGER_KEY:
            logs = {
                'train_loss': log_losses,
                'metrics': log_metrics
            }

        elif self.hparams.logger == TENSORBOARD_LOGGER_KEY:
            # in PL 0.8.1 can't log nested metrics
            # so need to flatten and group using slash syntax
            log_metrics = {'train/'+k: v for k,v in log_metrics.items()}
            logs = {
                'train/full_loss': log_losses,
                **log_metrics
            }
        else:
            logs = {'train_loss': log_losses}

        return logs

    def log_images(self, batch, output, batch_idx, phase):
        if self.hparams.logger == WANDB_LOGGER_KEY:
            images = wandb_prep_images(phase, batch, output, batch_idx, self.hparams.log_images_interval)
        elif self.hparams.logger == TENSORBOARD_LOGGER_KEY:
            images = tensorboard_prep_images(phase, batch, output, batch_idx, self.hparams.log_images_interval)
        else:
            images = {}

        return images

    def log_val_test(self, outputs, avg_metrics_values):
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

        return logs

    def on_train_end(self):
        self.trainer.checkpoint_callback._save_model(filepath=self.trainer.checkpoint_callback.dirpath+'/latest.ckpt')


    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.dataloaders.train.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=7,
                          drop_last=True, # to avoid batch_size=1
                          )

    def val_dataloader(self):
        # REQUIRED
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.dataloaders.val.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=7,
                          )

    def test_dataloader(self):
        # REQUIRED
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.dataloaders.test.batch_size,
                          shuffle=False,
                          pin_memory=False,
                          num_workers=7,
                          )