'''
python ./eval.py \
'/home/clear/fbartocc/output_data/depth_project/MonocularDepthKitti/outputs' \
'MD2_photo_scaled_smooth1e-3_lr1e-4_batch8_0321_174751,MD2_photo_scaled_smooth1e-3_lr1e-5_batch8_0308_170204' \
'/home/clear/fbartocc/working_data/KITTI/MC_sparse_lidar' \
'/home/clear/fbartocc/working_data/depth_project/KITTI/abn_semantic_segmentation' \
'/home/clear/fbartocc/output_data/depth_project/Reports_PostCVPR_KITTI/' \


python ./eval.py \
'/home/clear/fbartocc/output_data/depth_project/MonocularDepthKitti/outputs' \
$RUN_LIST \
'/home/clear/fbartocc/working_data/KITTI/MC_sparse_lidar' \
'/home/clear/fbartocc/working_data/depth_project/KITTI/abn_semantic_segmentation' \
'/home/clear/fbartocc/output_data/depth_project/Reports_PostCVPR_KITTI/'
'''


import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
from pathlib import Path
import torch
import numpy as np
import random
import click
from pprint import pprint
from tqdm import tqdm
from PIL import Image

from pytorch_lightning.utilities import move_data_to_device

import torch.nn.functional as F

from models.multi_view import MultiViewModel

from utils.eval_utils import (get_ckpt_path, load_model_trainer, get_best_ckt_name, untensor,
                              gen_augmented_inputs, gen_augmented_outputs)
from utils.common_logging import average_metrics
from utils.depth import inv2depth, compute_depth_metrics


def set_seed(seed):
    if seed >= 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


def batch_filename_to_semseg_path(filename):
    '''
    e.g., filename == '2011_09_26_0002_c2_0000000006'
    '''
    date = filename[:10]
    num_seq = filename[11:15]
    camera_idx = filename[17:18]
    idx = int(filename[-10:])

    return f'{date}/{date}_drive_{num_seq}_sync/image_0{camera_idx}/data/{idx:010d}.png'


def eval_with_outputs(model, dataloader, log_interval, brightness_factors, semantic_root_dir, collate_fn):
    list_of_metrics = []
    list_of_semseg_metrics = []
    outputs = []
    originals = []

    list_of_aug_metrics = {str(bf): [] for bf in brightness_factors}
    list_of_aug_metrics['lidar_shutdown'] = []
    augmented_outputs = []
    augmented_inputs = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(model.test_dataset) // dataloader.batch_size):
            batch = move_data_to_device(batch, 0)

            batch_idxs = [batch_idx for batch_idx in batch['idx']]
            batch_filenames = [batch_filename for batch_filename in batch['filename']]

            # Calculate vanilla predicted metrics
            sparse_lidar = batch.get('sparse_projected_lidar', None)
            sparse_pc = batch.get('sparse_lidar_pc', None)
            if sparse_pc is not None:
                sparse_pc = sparse_pc.to(batch['target_view'].device)

            inv_depths = model.compute_inv_depths(
                batch['target_view'],
                sparse_depth=sparse_lidar,
                sparse_pc=sparse_pc,
                intrinsics=batch['intrinsics']
            )['inv_depths']

            inv_depth = inv_depths[0]
            depth = inv2depth(inv_depth)

            metrics = compute_depth_metrics(gt=batch['projected_lidar'], pred=depth, **model.hparams.metrics)
            if metrics is not None:
                list_of_metrics.append(metrics)

            # Compute semseg filtered metrics
            batch_semgseg_relative_paths = [batch_filename_to_semseg_path(filename) for filename in batch_filenames]
            semsegs = [np.array(Image.open(semantic_root_dir / rel_path)) for rel_path in batch_semgseg_relative_paths]
            semsegs = torch.tensor(semsegs).to(batch['target_view'].device).unsqueeze(0)
            semsegs = F.interpolate(semsegs, size=batch['target_view'].shape[-2:], mode='nearest')
            mask = (semsegs == 55).squeeze()  # True for pixel of cars False otherwise
            metrics = compute_depth_metrics(gt=batch['projected_lidar'], pred=depth, mask=mask, **model.hparams.metrics)
            if metrics is not None:
                list_of_semseg_metrics.append(metrics)

            # Calculate augmented predicted metrics
            aug_inputs = gen_augmented_inputs(brightness_factors, model.test_dataset,
                                              batch_idxs, batch_filenames, collate_fn)
            aug_outputs = gen_augmented_outputs(aug_inputs, model)

            for filename, tuple_outputs in aug_outputs.items():
                brightness_adjusted_outputs, lidar_shutdown_output = tuple_outputs

                for i, aug_inv_depth in enumerate(brightness_adjusted_outputs):
                    depth = inv2depth(aug_inv_depth)
                    metrics = compute_depth_metrics(gt=batch['projected_lidar'], pred=depth, **model.hparams.metrics)
                    if metrics is not None:
                        list_of_aug_metrics[str(brightness_factors[i])].append(metrics)

                depth = inv2depth(lidar_shutdown_output)
                metrics = compute_depth_metrics(gt=batch['projected_lidar'], pred=depth, **model.hparams.metrics)
                if metrics is not None:
                    list_of_aug_metrics['lidar_shutdown'].append(metrics)

            if batch_idx % log_interval == 0:
                outputs.append(inv_depth)
                originals.append(batch)

                for filename, tuple_inputs in aug_inputs.items():
                    brightness_adjusted_x, lidar_shutdown_x = tuple_inputs
                    brightness_adjusted_x = [untensor(o['target_view'], to_int=True) for o in brightness_adjusted_x]
                    lidar_shutdown_x = untensor(lidar_shutdown_x['sparse_projected_lidar'], to_int=False)[:, :, 0]
                    augmented_inputs.append((brightness_adjusted_x, lidar_shutdown_x))

                for filename, tuple_outputs in aug_outputs.items():
                    brightness_adjusted_outputs, lidar_shutdown_output = tuple_outputs
                    brightness_adjusted_outputs = [untensor(o, to_int=False)[:, :, 0] for o in
                                                   brightness_adjusted_outputs]
                    lidar_shutdown_output = untensor(lidar_shutdown_output, to_int=False)[:, :, 0]
                    augmented_outputs.append((brightness_adjusted_outputs, lidar_shutdown_output))

    avg_metrics_values = average_metrics(list_of_metrics, prefix='test')
    avg_metrics_values = {k: float(v.cpu().numpy()) for k, v in avg_metrics_values.items()}

    avg_semseg_metrics_values = average_metrics(list_of_semseg_metrics, prefix='test/semseg')
    avg_semseg_metrics_values = {k: float(v.cpu().numpy()) for k, v in avg_semseg_metrics_values.items()}
    avg_metrics_values.update(avg_semseg_metrics_values)

    for augmentation_name, aug_metrics in list_of_aug_metrics.items():
        avg_aug_metrics_values = average_metrics(aug_metrics, prefix=f'test/{augmentation_name}')
        avg_aug_metrics_values = {k: float(v.cpu().numpy()) for k, v in avg_aug_metrics_values.items()}
        avg_metrics_values.update(avg_aug_metrics_values)

    inv_depths = [untensor(inv_depth, to_int=False)[:, :, 0] for inv_depth in outputs]
    cams = [untensor(batch['target_view']) for batch in originals]
    sparse_lidars = [untensor(batch['sparse_projected_lidar'], to_int=False)[:, :, 0] for batch in originals]
    gt_depths = [untensor(batch['projected_lidar'], to_int=False)[:, :, 0] for batch in originals]
    filenames = [batch['filename'][0] for batch in originals]
    idxs = [batch['idx'][0] for batch in originals]

    return {
        'results': avg_metrics_values,
        'inv_depths': inv_depths,
        'cam_images': cams,
        'sparse_lidars': sparse_lidars,
        'gt_depths': gt_depths,
        'filenames': filenames,
        'idxs': idxs,
        'augmented_inputs': augmented_inputs,
        'augmented_outputs': augmented_outputs,
    }


@click.command()
@click.argument('checkpoint_base_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('run_dir_names', type=str)
@click.argument('sparse_data_root_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('semantic_root_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('report_root_dir', type=click.Path(file_okay=False))
@click.option('--gt_scaled', is_flag=True)
@click.option('--eval_on_rawlidar', is_flag=True)
@click.option('--input_rawlidar', is_flag=True)
@click.option('--cmap', type=str, default='inferno')
@click.option('--batch_size', type=int, default=1)
@click.option('--nb_of_images_to_save', type=int, default=9)
@click.option('--geometric_dataloader', is_flag=True)
def main(checkpoint_base_dir, run_dir_names, sparse_data_root_dir, semantic_root_dir, report_root_dir, gt_scaled,
         eval_on_rawlidar, input_rawlidar, cmap, batch_size, nb_of_images_to_save, geometric_dataloader):

    print('Configuring eval....')

    if geometric_dataloader:
        from models.geometric_dataloader import DataLoader, Collater
        collate_fn = Collater()
    else:
        from torch.utils.data import DataLoader
        from torch.utils.data.dataloader import default_collate as collate_fn

    brightness_factors = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    test_img_size = [384, 1280]
    dpi = mpl.rcParams['figure.dpi']
    height, width = test_img_size
    scale = 2.0
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi) * scale, height / float(dpi) * scale
    print(figsize)


    str_gt_scaled = 'GTscaled' if gt_scaled else 'NOTscaled'
    if eval_on_rawlidar:
        split_file_path = '${data_split_dir}/eigen_test_files.txt'
        str_eval_on_raw = 'sparseRaw'
    else:
        split_file_path = '${data_split_dir}/filtered_eigen_test_files.txt'
        str_eval_on_raw = 'denseGT'
    # split_file_path = '${data_split_dir}/depth_completion_val.txt'

    str_lidar_input = '_RawInput' if input_rawlidar else '_4beamsInput'
    report_dir_name = \
        f'eval_on_{test_img_size[0]}x{test_img_size[1]}_{str_eval_on_raw}_{str_gt_scaled}{str_lidar_input}'
    report_dir = Path(report_root_dir) / report_dir_name
    report_dir.mkdir(mode=0o0700, parents=True, exist_ok=True)


    print('Evaluating given runs....')
    print("CHECKPOINT BASE DIR: ", checkpoint_base_dir)
    set_seed(0)

    checkpoint_base_dir = Path(checkpoint_base_dir)
    run_list = sorted(run_dir_names.split(','))

    ckpt_names = sorted([d.stem for d in (checkpoint_base_dir / run_list[0] / 'version_0').iterdir()])
    best_ckpt_name = get_best_ckt_name(ckpt_names)
    ckpt_path = get_ckpt_path(checkpoint_base_dir, run_list[0], ckpt_name=best_ckpt_name)
    model, _ = load_model_trainer(MultiViewModel, ckpt_path)
    test_dataset_config = copy.deepcopy(model.hparams.datasets.test)
    print('test_dataset_config:')
    print(test_dataset_config)

    sparse_data_root_dir = Path(sparse_data_root_dir)
    if not input_rawlidar: # input is 4 beam
        if eval_on_rawlidar:
            test_dataset_config.gt_depth_root_dir = str(sparse_data_root_dir / 'factor_0')
        test_dataset_config.sparse_depth_root_dir = str(sparse_data_root_dir / 'factor_16')
    else:
        test_dataset_config.eval_on_sparse = eval_on_rawlidar
        test_dataset_config.sparse_depth_root_dir = str(sparse_data_root_dir / 'factor_0')

    # test_dataset_config.gt_depth_root_dir = test_dataset_config.sparse_depth_root_dir
    test_dataset_config.split_file_path = split_file_path
    test_dataset_config.eval_on_sparse = False
    test_dataset_config.data_transform_options.image_shape = test_img_size

    from dataloaders.kitti import SequentialKittiLoader
    from dataloaders.transforms import test_transforms

    test_dataset = SequentialKittiLoader(**test_dataset_config, data_transform=test_transforms, input_channels=3)

    log_interval = len(test_dataset) // (nb_of_images_to_save * batch_size)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=0,
                                 worker_init_fn=set_seed
                                 )

    semantic_root_dir = Path(semantic_root_dir)

    result_file = report_dir / 'perfs.csv'

    for i, run_name in enumerate(run_list):

        print('*' * 90)
        print('=' * 5, run_name, '=' * 5)
        print('*' * 60)

        torch.cuda.empty_cache()

        set_seed(0)

        run_dir = checkpoint_base_dir / run_name / 'version_0'
        ckpt_names = sorted([d.stem for d in run_dir.iterdir()])
        print('ckpt_names:')
        pprint(ckpt_names)
        best_ckpt_name = get_best_ckt_name(ckpt_names, max_epoch=None)
        print('best_ckpt_name: ', best_ckpt_name)
        ckpt_path = get_ckpt_path(checkpoint_base_dir, run_name, ckpt_name=best_ckpt_name)
        model, _ = load_model_trainer(MultiViewModel, ckpt_path)

        model.test_dataset = test_dataset

        model.hparams.logger = None
        model.hparams.metrics = {
            'crop': 'garg',
            'min_depth': 0.001,
            'max_depth': 80.,
            'use_gt_scale': True
        }

        model.eval()
        model.cuda()

        model.hparams.metrics.use_gt_scale = gt_scaled

        outputs = eval_with_outputs(model, test_dataloader, log_interval, brightness_factors,
                                    semantic_root_dir, collate_fn)

        if i == 0:

            if result_file.exists() and result_file.stat().st_size > 0:
                cols = [name.split('/')[1] for name in outputs['results'].keys()]
                cols = ['run_name'] + cols
                cols = '\t'.join(cols)
                cols += '\n'
                with open(str(result_file), 'a') as f:
                    f.write(cols)

            cam_images_dir = report_dir / 'cam_images'
            cam_images_dir.mkdir(mode=0o0700, parents=True, exist_ok=True)
            sparse_lidar_dir = report_dir / 'sparse_lidar'
            sparse_lidar_dir.mkdir(mode=0o0700, parents=True, exist_ok=True)
            gt_depths_dir = report_dir / 'gt_depths'
            gt_depths_dir.mkdir(mode=0o0700, parents=True, exist_ok=True)

            for file_name, img in zip(outputs['filenames'], outputs['cam_images']):
                plt.figure(figsize=figsize)
                plt.imshow(img)
                plt.gca().axis('off')
                plt.savefig(str(cam_images_dir / f'{file_name}.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
                plt.close()

            for file_name, img in zip(outputs['filenames'], outputs['sparse_lidars']):
                plt.figure(figsize=figsize)
                plt.imshow(img, cmap=cmap)
                plt.gca().axis('off')
                plt.savefig(str(sparse_lidar_dir / f'{file_name}.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
                plt.close()

            for file_name, img in zip(outputs['filenames'], outputs['gt_depths']):
                plt.figure(figsize=figsize)
                plt.imshow(img, cmap=cmap)
                plt.gca().axis('off')
                plt.savefig(str(gt_depths_dir / f'{file_name}.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
                plt.close()

            for file_name, tuple_inputs in zip(outputs['filenames'], outputs['augmented_inputs']):
                for j, brightness_aug_img in enumerate(tuple_inputs[0]):
                    brightness_factor_dir = cam_images_dir / f'brightness_factor={brightness_factors[j]}'
                    brightness_factor_dir.mkdir(mode=0o0700, parents=True, exist_ok=True)

                    plt.figure(figsize=figsize)
                    plt.imshow(brightness_aug_img)
                    plt.gca().axis('off')
                    plt.savefig(str(brightness_factor_dir / f'{file_name}.png'), dpi=dpi, bbox_inches='tight',
                                pad_inches=0)
                    plt.close()

        results = [f'{val:.4f}' for val in outputs['results'].values()]
        line = [run_name] + results
        line = '\t'.join(line)
        line += '\n'

        with open(str(result_file), 'a') as f:
            f.write(line)

        run_report_dir = report_dir / run_name

        vanilla_dir = run_report_dir / 'vanilla'
        vanilla_dir.mkdir(mode=0o0700, parents=True, exist_ok=True)
        luminosity_dir = run_report_dir / 'luminosity'
        luminosity_dir.mkdir(mode=0o0700, parents=True, exist_ok=True)
        lidar_shutdown_dir = run_report_dir / 'lidar_shutdown'
        lidar_shutdown_dir.mkdir(mode=0o0700, parents=True, exist_ok=True)

        for file_name, img in zip(outputs['filenames'], outputs['inv_depths']):
            plt.figure(figsize=figsize)
            plt.imshow(img, cmap='inferno')
            plt.gca().axis('off')
            plt.savefig(str(vanilla_dir / f'{file_name}.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()

        for file_name, augmented_output in zip(outputs['filenames'], outputs['augmented_outputs']):

            # plot lidar_shutdown
            plt.figure(figsize=figsize)
            plt.imshow(augmented_output[1], cmap=cmap)
            plt.gca().axis('off')
            plt.savefig(str(lidar_shutdown_dir / f'{file_name}.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()

            # plot brightness aug
            for j, brightness_output in enumerate(augmented_output[0]):
                brightness_factor_dir = luminosity_dir / f'brightness_factor={brightness_factors[j]}'
                brightness_factor_dir.mkdir(mode=0o0700, parents=True, exist_ok=True)

                plt.figure(figsize=figsize)
                plt.imshow(brightness_output, cmap=cmap)
                plt.gca().axis('off')
                plt.savefig(str(brightness_factor_dir / f'{file_name}.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
                plt.close()

if __name__ == '__main__':
    main()