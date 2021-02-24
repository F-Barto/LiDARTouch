"""
This file runs the main training/val loop, etc... using Lightning Trainer

more details at:
 https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.html#training-loop-structure
"""

from argparse import ArgumentParser
from pathlib import Path
import torch
import numpy as np
import random

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger

from models.multi_view import MultiViewModel
from models.supervised import FullySupervisedModel

from utils.config import load_yaml


def set_random_seed(seed):
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(project_config, hparams, resume=False):
    torch.manual_seed(0)
    np.random.seed(0)

    # init module
    models = {
        'multi_view': MultiViewModel,
        'supervised': FullySupervisedModel
    }
    assert hparams.model in models
    model = models[hparams.model](hparams)

    # tags associated to the run
    def shape_format(shape):
        # shape = [Height, Width]
        return f"{shape[1]}x{shape[0]}"

    #assert hparams.metrics.use_gt_scale != hparams.datasets.train.load_pose, f"Either velocity of gt scaled"


    base_output_dir = Path(project_config.output_dir) / project_config.project_name

    logs_dir = base_output_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    experiment_output_dir = base_output_dir / 'outputs' / project_config.experiment_name
    latest_ckpt = experiment_output_dir / 'version_0/last.ckpt'

    if resume and latest_ckpt.exists():
        model = model.load_from_checkpoint(str(latest_ckpt))
        trainer = Trainer(resume_from_checkpoint=str(latest_ckpt))
        trainer.fit(model)
        trainer.test(model)

    else:

        if hparams.get('model_ckpt', None) is not None:
            saved_state_dict = torch.load(hparams.model_ckpt, map_location='cpu')['state_dict']
            model.load_state_dict(saved_state_dict)

        assert hparams.logger in ['wandb', 'tensorboard']

        if hparams.logger == 'tensorboard':
            experiment_logger = TensorBoardLogger(
                save_dir=logs_dir,
                name=project_config.experiment_name
            )

            run_output_dir = experiment_output_dir / f'version_{experiment_logger.version}'

        elif hparams.logger == 'wandb':

            list_of_tags = [
                f"{hparams.network.depth_net.name} DepthNet",
                f"{hparams.network.pose_net.name} PoseNet",
                hparams.optimizer.name,
                hparams.scheduler.name,
                {1: 'gray', 3: 'rgb'}[hparams.input_channels],
                f"train-{shape_format(hparams.datasets.train.data_transform_options.image_shape)}",
                f"val-{shape_format(hparams.datasets.val.data_transform_options.image_shape)}",
                f"test-{shape_format(hparams.datasets.test.data_transform_options.image_shape)}",
            ]
            if project_config.mixed_precision:
                list_of_tags.append('mixed_precision')

            losses = list(hparams.losses.keys())
            if 'supervised_loss_weight' in losses:
                losses.remove('supervised_loss_weight')
            list_of_tags += losses

            experiment_logger = WandbLogger(
                project = project_config.project_name,
                save_dir=logs_dir, # the path to a directory where artifacts will be written
                log_model=True,
                tags=list_of_tags
            )
            #wandb_logger.watch(model, log='all', log_freq=5000) # watch model's gradients and params

            run_output_dir = experiment_output_dir / f'version_{experiment_logger.experiment.id}'

        else:
            run_output_dir = experiment_output_dir / 'no_version_system'

        run_output_dir.mkdir(parents=True, exist_ok=True)
        run_output_dir = str(run_output_dir)

        checkpoint_callback = ModelCheckpoint(
            filepath=run_output_dir + '/{epoch:04d}-{val-rmse_log:.5f}', # saves a file like: my/path/epoch=2-abs_rel=0.0115.ckpt
            save_top_k=15,
            save_last=True,
            verbose=True,
            monitor='val-rmse_log',
            mode='min',
        )

        lr_logger = LearningRateLogger()


        if project_config.mixed_precision:
            amp_level='01'
            precision=16

        if project_config.gpus > 1:
            distributed_backend = 'ddp'
        else:
            distributed_backend = None

        profiler = False
        if project_config.fast_dev_run:
            from pytorch_lightning.profiler import AdvancedProfiler
            profiler = AdvancedProfiler(output_filename='./profiler.log')


        trainer = Trainer(
            gpus=project_config.gpus,
            distributed_backend=distributed_backend,
            num_nodes =project_config.nodes,
            checkpoint_callback=checkpoint_callback,
            callbacks=[lr_logger],
            logger=experiment_logger,
            fast_dev_run=project_config.fast_dev_run,
            profiler=profiler,
            early_stop_callback=False,
            limit_train_batches=hparams.limit_train_batches,
            #amp_level='O1',
            #precision=16,
            **hparams.trainer
        )
        trainer.fit(model)
        trainer.test(model)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_root', '-cr', type=str, required=True)
    parser.add_argument('--project_config_file', '-pf', type=str, required=True)
    parser.add_argument('--project_config_overrides', '-po', type=str)
    parser.add_argument('--model_config_file', '-mf', type=str, required=True)
    parser.add_argument('--model_config_overrides', '-mo', type=str)
    parser.add_argument('--resume', action='store_true')
    # parse params
    args = parser.parse_args()

    import os
    print("OAR_JOB_ID:", os.environ['OAR_JOB_ID'])

    print()
    print('=' * 30, " PROJECT CONFIG", '=' * 30)

    po = list(args.project_config_overrides.split())
    print()
    print('-' * 30, " overrides", '-' * 30)
    print(po)

    project_config = load_yaml(args.project_config_file, args.config_root, overrides= po)
    print()
    print('-' * 30, " config", '-' * 30)
    print(project_config.pretty())

    print()
    print('=' * 30, " MODEL CONFIG", '=' * 30)
    mo = list(args.model_config_overrides.split())
    print()
    print('-' * 30, " overrides", '-' * 30)
    print(mo)

    hparams = load_yaml(args.model_config_file, args.config_root, overrides= mo)
    print()
    print('-' * 30, " config", '-' * 30)
    print(hparams.pretty())

    print()
    print('='*80)

    set_random_seed(hparams.seed)

    main(project_config, hparams, args.resume)