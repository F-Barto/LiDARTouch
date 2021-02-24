from matplotlib.cm import get_cmap
from utils.types import is_tensor, is_dict
import numpy as np

try:
    import wandb
    from wandb.wandb_run import Run
except ImportError:
    raise ImportError('You want to use `wandb` logger which is not installed yet,' 
                      ' install it with `pip install wandb`.')

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

def prep_image(key, image, caption=None):
    """
    Prepare image for wandb logging
    Parameters
    ----------
    key : str
        Key from data containing the inverse depth map
    image : torch.Tensor [3,H,W]
        Image to be logged
    Returns
    -------
    output : dict
        Dictionary with key and value for logging
    """
    if is_tensor(image):
        image = image.detach().permute(1, 2, 0).cpu().numpy()
    caption = key if caption is None else caption
    return wandb.Image(image, caption=caption)

def log_rgb(key, batch, i=0, caption=None):
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
    image : wandb.Image
        Wandb image ready for logging
    """
    rgb = batch[key] if is_dict(batch) else batch
    return prep_image(key, rgb[i], caption=caption)

def log_depth(key, batch, i=0, caption=None):
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
    image : wandb.Image
        Wandb image ready for logging
    """
    depth = batch[key] if is_dict(batch) else batch
    inv_depth = 1. / depth[i]
    inv_depth[depth[i] == 0] = 0

    # converts a depth map to a colormap for visualization.
    depth_colormapped = viz_inv_depth(inv_depth, filter_zeros=True)

    return prep_image(key, depth_colormapped, caption=caption)

def log_inv_depth(key, batch, i=0, caption=None):
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
    image : wandb.Image
        Wandb image ready for logging
    """
    inv_depth = batch[key] if is_dict(batch) else batch
    # converts an inverse depth map to a colormap for visualization.
    inv_depth_colormapped = viz_inv_depth(inv_depth[i])

    return prep_image(key, inv_depth_colormapped, caption=caption)

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
            A dict Wandb image (wandb.Image) ready for logging
        """

    if batch_idx % log_images_interval != 0:
        return {}

    i = 0 # for each batch, we always log the first image of the batch only

    prefix = f"{learning_phase}-{batch['filename'][i]}-batch{batch_idx}"

    img_list = [
        log_rgb('target_view', batch, i=i),
        log_inv_depth('inv_depth', output, i=i),
        log_depth('depth', output, i=i, caption='pred_depth'),
        log_depth('projected_lidar', batch, i=i, caption='lidar_depth')
    ]

    if hasattr(batch, 'sparse_projected_lidar'):
        sparse_projected_lidar = log_depth('sparse_projected_lidar', batch, i=i, caption='sparse_lidar')
        img_list.append(sparse_projected_lidar)

    return {prefix: img_list}
