import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import itertools

import torchvision.transforms.functional as TF
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import move_data_to_device

from lidartouch.utils.depth import inv2depth


################################################
################# VISUALIZATION ################
################################################

def gridplot(imgs, titles=[], cmaps=[], cols=2, figsize=(12, 12), wspace=None, hspace=None, show=False, vmax=None):
    """
    Plot a list of images in a grid format

    :param imgs: list of images to plot
    :param titles: list of titles to print above each image of `imgs`
    :param cols: number of column of the grid, the number of rows is determined accordingly
    :param figsize: matplotlib `figsize` figure param
    """

    rows = len(imgs) // cols + len(imgs) % cols

    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    if len(imgs) <= 1:
        axs.imshow(imgs[0], cmap=cmaps[0], vmax=vmax)
        axs.set_title(titles[0])
        axs.axis('off')
    else:
        axs = axs.flatten()

        for img, title, cmap, ax in itertools.zip_longest(imgs, titles, cmaps, axs):

            if img is None:
                ax.set_visible(False)
                continue

            ax.imshow(img, cmap=cmap, vmax=vmax)
            ax.set_title(title)
            ax.axis('off')

    if wspace is None or hspace is None:
        fig.tight_layout()
    else:
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

    if show:
        plt.show()
    else:
        return fig


def untensor(img, to_int=True):
    if len(img.shape) > 3:
        img = img[0]

    if img.get_device() >= 0:
        img = img.cpu()

    img = img.numpy()

    if to_int:
        img = (img * 255).astype(int)

    return img.transpose(1, 2, 0)


def plot_results(x, outputs, figsize=(10, 7), pred_key='inv_depths', depth_cmap='inferno', split_name='val'):
    # depth = viz_inv_depth(outputs[pred_key][0][0])
    depth = untensor(inv2depth(outputs[pred_key][0])[0], to_int=False)[:, :, 0]

    target_view = untensor(x['target_view'])
    sparse_lidar = untensor(x['sparse_projected_lidar'], to_int=False)[:, :, 0]

    imgs = [target_view, sparse_lidar, depth]
    cmaps = [None, depth_cmap, depth_cmap]
    titles = ['target_view', 'sparse_lidar', 'depth']

    cols = 1

    if not 'train' in split_name:
        gt_depth = untensor(x['projected_lidar'], to_int=False)[:, :, 0]
        imgs.append(gt_depth)
        cmaps.append(depth_cmap)
        titles.append('gt_depth')

    gridplot(imgs, titles=titles, cols=cols, cmaps=cmaps, figsize=figsize, show=True)


################################################
############## ADVERSIAL EXEMPLE ###############
################################################


def adjust_sample_brightness(x, brightness_factor=2.0):
    arr = (x['target_view'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = TF.adjust_brightness(img, brightness_factor)

    transform = transforms.ToTensor()
    tensor_img = transform(img)

    x['target_view'] = tensor_img

    return x


def lidar_shutdown(x):
    zeros = torch.zeros(x['sparse_projected_lidar'].shape)
    x['sparse_projected_lidar'] = zeros

    return x


def gen_augmented_inputs(brightness_factors, dataset, batch_idxs, filenames, collate_fn):
    inputs = {}

    for batch_idx, filename in zip(batch_idxs, filenames):
        sample = dataset[batch_idx]

        brightness_adjusted_samples = [adjust_sample_brightness(sample.copy(), brightness_factor=b) for b in
                                       brightness_factors]

        lidar_shutdown_sample = lidar_shutdown(sample.copy())

        lidar_shutdown_x = collate_fn([lidar_shutdown_sample])
        brightness_adjusted_x = [collate_fn([brightness_adjusted_sample]) for brightness_adjusted_sample in
                                 brightness_adjusted_samples]

        inputs[filename] = (brightness_adjusted_x, lidar_shutdown_x)

    return inputs



def gen_augmented_inputs_from_outputs(brightness_factors, dataset, outputs, collate_fn):
    inputs = {}

    for sample_idx, filename in zip(outputs['idxs'], outputs['filenames']):
        sample = dataset[sample_idx]

        brightness_adjusted_samples = [adjust_sample_brightness(sample.copy(), brightness_factor=b) for b in
                                       brightness_factors]

        lidar_shutdown_sample = lidar_shutdown(sample.copy())

        lidar_shutdown_x = collate_fn([lidar_shutdown_sample])
        brightness_adjusted_x = [collate_fn([brightness_adjusted_sample]) for brightness_adjusted_sample in
                                 brightness_adjusted_samples]

        inputs[filename] = (brightness_adjusted_x, lidar_shutdown_x)

    return inputs


def gen_augmented_outputs(inputs, model, out_for_viz=False):
    augmented_outputs = {}

    for filename, tuple_inputs in inputs.items():

        brightness_adjusted_xs, lidar_shutdown_xs = tuple_inputs

        brightness_adjusted_outputs = []

        for brightness_adjusted_x in brightness_adjusted_xs:

            brightness_adjusted_x = move_data_to_device(brightness_adjusted_x, 0)

            with torch.no_grad():

                sparse_lidar = brightness_adjusted_x.get('sparse_projected_lidar', None)
                sparse_pc = brightness_adjusted_x.get('sparse_lidar_pc', None)
                if sparse_pc is not None:
                    sparse_pc = sparse_pc.to(brightness_adjusted_x['target_view'].device)

                brightness_adjusted_inv_depths = model.compute_inv_depths(
                    brightness_adjusted_x['target_view'],
                    sparse_depth=sparse_lidar,
                    sparse_pc=sparse_pc,
                    intrinsics=brightness_adjusted_x['intrinsics']
                )

                brightness_adjusted_inv_depths = brightness_adjusted_inv_depths['inv_depths']

            brightness_adjusted_outputs.append(brightness_adjusted_inv_depths[0])

            torch.cuda.empty_cache()

        lidar_shutdown_x = move_data_to_device(lidar_shutdown_xs, 0)
        with torch.no_grad():
            sparse_lidar = lidar_shutdown_x.get('sparse_projected_lidar', None)
            sparse_pc = lidar_shutdown_x.get('sparse_lidar_pc', None)
            if sparse_pc is not None:
                sparse_pc = sparse_pc.to(lidar_shutdown_x['target_view'].device)

            lidar_shutdown_inv_depths = model.compute_inv_depths(
                lidar_shutdown_x['target_view'],
                sparse_depth=sparse_lidar,
                sparse_pc=sparse_pc,
                intrinsics=lidar_shutdown_x['intrinsics']
            )

            lidar_shutdown_inv_depths = lidar_shutdown_inv_depths['inv_depths']

        lidar_shutdown_output = lidar_shutdown_inv_depths[0]

        if out_for_viz:
            brightness_adjusted_outputs = [untensor(o)[:, :, 0] for o in brightness_adjusted_outputs]
            lidar_shutdown_output = untensor(lidar_shutdown_output)[:, :, 0]

        augmented_outputs[filename] = (brightness_adjusted_outputs, lidar_shutdown_output)

    return augmented_outputs


################################################
################# MODEL LOADING ################
################################################

def get_ckpt_path(checkpoint_base_dir, run_name, ckpt_name='last'):
    run_dir = checkpoint_base_dir / run_name / 'version_0'
    ckpt_path = run_dir / (ckpt_name + '.ckpt')

    return ckpt_path


def load_model_trainer(model_cls, ckpt_path):
    model = model_cls.load_from_checkpoint(str(ckpt_path))

    trainer = Trainer(resume_from_checkpoint=str(ckpt_path), gpus=1)

    return model, trainer


def get_best_ckt_name(ckpt_names, max_epoch=None):
    '''

    max_epoch: ensure that returned checkpoint has been trained for fewer epochs than max_epoch
    '''
    min_val = None
    min_ckpt_name = 'last.ckpt'
    for ckpt_name in ckpt_names:
        if 'epoch' in ckpt_name:
            epoch = int(ckpt_name.split('=')[1][:4])
            epoch = True if max_epoch is None else epoch <= max_epoch
            metric_val = float(ckpt_name.split('=')[-1])
            min_val = metric_val if min_val is None or metric_val <= min_val else min_val
            min_ckpt_name = ckpt_name if min_val and epoch else min_ckpt_name
    return min_ckpt_name


