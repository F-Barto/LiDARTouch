from matplotlib.cm import get_cmap
from utils.types import is_tensor, is_dict
import numpy as np
import matplotlib.pyplot as plt
import itertools

def gridplot(imgs, titles=[], cmaps=[], cols=2, figsize=(12, 12)):
    """
    Plot a list of images in a grid format

    :param imgs: list of images to plot
    :param titles: list of titles to print above each image of `imgs`
    :param cols: number of column of the grid, the number of rows is determined accordingly
    :param figsize: matplotlib `figsize` figure param
    """

    rows =  len(imgs) // cols + len(imgs) % cols

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()

    for img, title, cmap, ax in itertools.zip_longest(imgs, titles, cmaps, axs):

        if img is None:
            ax.set_visible(False)
            continue

        if img.ndim == 2 and cmap is None:
            cmap = 'gray'

        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    return fig

def viz_inv_depth(inv_depth, normalizer=None, percentile=95,
                  colormap='plasma', filter_zeros=False):
    """
    Converts an inverse depth map to a colormap for visualization.
    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map to be converted
    normalizer : float
        Value for inverse depth map normalization
    percentile : float
        Percentile value for automatic normalization
    colormap : str
        Colormap to be used
    filter_zeros : bool
        If True, do not consider zero values during normalization
    Returns
    -------
    colormap : np.array [H,W,3]
        Colormap generated from the inverse depth map
    """
    # If a tensor is provided, convert to numpy
    if is_tensor(inv_depth):
        # Squeeze if depth channel exists
        if len(inv_depth.shape) == 3:
            inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth /= (normalizer + 1e-6)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]

def prep_image(image):
    """
    Prepare image for tensorboard logging
    Parameters
    ----------
    image : torch.Tensor [3,H,W]
        Image to be logged
    Returns
    -------
    image : numpy array
        image as numpy array
    """
    if is_tensor(image):
        image = image.detach().permute(1, 2, 0).cpu().numpy()

    if image.shape[-1] == 1:
        return image[:,:,0]
    return image

def prep_rgb(key, batch, i=0):
    """
    Converts an RGB image from a batch for logging
    Parameters
    ----------
    key : str
        Key from data containing the image
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the image
    Returns
    -------
    image : numpy array
        image as numpy array
    """
    rgb = batch[key] if is_dict(batch) else batch
    return prep_image(rgb[i])

def prep_depth(key, batch, i=0):
    """
    Converts a depth map from a batch for logging
    Parameters
    ----------
    key : str
        Key from data containing the depth map
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the depth map
    Returns
    -------
    image : numpy array
        image as numpy array
    """
    depth = batch[key] if is_dict(batch) else batch
    inv_depth = 1. / depth[i]
    inv_depth[depth[i] == 0] = 0

    # converts a depth map to a colormap for visualization.
    depth_colormapped = viz_inv_depth(inv_depth, filter_zeros=True)

    return depth_colormapped

def prep_inv_depth(key, batch, i=0):
    """
    Converts an inverse depth map from a batch for logging
    Parameters
    ----------
    key : str
        Key from data containing the inverse depth map
    batch : dict
        Dictionary containing the key
    i : int
        Batch index from which to get the inverse depth map
    Returns
    -------
    image : numpy array
        image as numpy array
    """
    inv_depth = batch[key] if is_dict(batch) else batch
    # converts an inverse depth map to a colormap for visualization.
    inv_depth_colormapped = viz_inv_depth(inv_depth[i])

    return inv_depth_colormapped

def prepare_images_to_log(learning_phase, batch, output, batch_idx, log_images_interval):
    """
        Take an input batch a prediction, converts the depth and inverse maps to a colormap for visualization,
        and  generates a dict of Wandb images ready for logging for orignal image, depth and inverse depth maps.

        Parameters
        ----------
        learning_phase : str
            Either 'train', 'val' or 'test'
        batch : dict
            Dictionary from the dataloader assumes to have the fields: depth, target_view, and filename
        inv_depth : torch.Tensor [B,1,H,W]
            Inverse depth maps prediction
        batch_idx: int
            Index of the batch
        Returns
        -------
        images : dict
            A dict with a matplotlib figure ready for tensorboard logging
        """

    if batch_idx % log_images_interval != 0:
        return {}

    i = 0 # for each batch, we always log the first image of the batch only

    prefix = f"{learning_phase}/{batch['filename'][i]}-batch{batch_idx}"

    img_list = [
        prep_rgb('target_view', batch, i=i),
        prep_inv_depth('inv_depth', output, i=i),
        prep_depth('projected_lidar', batch, i=i)
    ]

    titles = ['target_view', 'inv_depth', 'gt_depth']

    cmaps = [None, 'magma', 'magma']

    if batch.get('sparse_projected_lidar', None) is not None:
        sparse_projected_lidar = prep_depth('sparse_projected_lidar', batch, i=i)
        img_list.append(sparse_projected_lidar)
        titles.append('sparse_lidar')
        cmaps.append('magma')

    if 'cam_disp' in output:
        img_list.append(prep_inv_depth('cam_disp', output, i=i))
        titles.append('cam_disp')
        cmaps.append('magma')

    if 'lidar_disp' in output:
        img_list.append(prep_inv_depth('lidar_disp', output, i=i))
        titles.append('lidar_disp')
        cmaps.append('magma')

    if 'uncertainties' in output:
        img_list.append(prep_inv_depth('uncertainties', output, i=i))
        titles.append('uncertainties')
        cmaps.append('magma')




    plt_figure = gridplot(img_list, titles=titles, cmaps=cmaps, cols=4, figsize=(6*4, 6*((len(titles)+1)//4)))

    return {prefix: plt_figure}

