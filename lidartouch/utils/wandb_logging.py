from matplotlib.cm import get_cmap
from lidartouch.utils.types import is_tensor, is_dict
import numpy as np

try:
    import wandb
except ImportError:
    raise ImportError('You want to use `wandb` logger which is not installed yet,' 
                      ' install it with `pip install wandb`.')

def get_maybe_from_dict(input, key):
    return input[key] if is_dict(input) else input

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

def prep_rgb(key, input, i=0, caption=None):
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
    rgb = get_maybe_from_dict(input, key)
    return prep_image(key, rgb[i], caption=caption)

def prep_depth(key, input, i=0, caption=None):
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
    depth = get_maybe_from_dict(input, key)
    inv_depth = 1. / depth[i]
    inv_depth[depth[i] == 0] = 0

    # converts a depth map to a colormap for visualization.
    depth_colormapped = viz_inv_depth(inv_depth, filter_zeros=True)

    return prep_image(key, depth_colormapped, caption=caption)

def prep_inv_depth(key, input, i=0, caption=None):
    """
    Converts an inverse depth map from a batch for logging
    Parameters
    ----------
    key : str
        Key from data containing the inverse depth map
    input : dict
        Element to log or dictionary containing the element to log with key ``key`
    i : int
        Batch index from which to get the inverse depth map
    Returns
    -------
    image : wandb.Image
        Wandb image ready for logging
    """
    inv_depth = get_maybe_from_dict(input, key)
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

    if learning_phase == 'train':
        prefix = f"{learning_phase}-batch{batch_idx}"
    else: #val, test
        prefix = f"{learning_phase}-{batch['filename'][i]}-batch{batch_idx}"

    img_list = [prep_rgb('target_view', batch, i=i)]


    if 'inv_depth' in output:
        img_list.append(prep_inv_depth('inv_depth', output, i=i))

    if 'inv_depths' in output:
        inv_depths = output['inv_depths']
        for scale_idx, inv_depth in enumerate(inv_depths):
            img_list.append(prep_inv_depth(f'inv_depth_{scale_idx}', inv_depth, i=i))

    if 'gt_depth' in batch:
        img_list.append(prep_depth('gt_depth', batch, i=i, caption='gt_depth'))

    if 'sparse_depth' in batch:
        img_list.append(prep_depth('sparse_depth', batch, i=i, caption='sparse_depth'))


    if 'outputs_by_scales' in output:
        outputs_by_scales = output['outputs_by_scales']

        img_list += maybe_prep_loss_output(outputs_by_scales, ['estimated', 'reconstructions'], i=i)
        img_list += maybe_prep_loss_output(outputs_by_scales, ['estimated', 'reconstruction_losses'], i=i)
        img_list += maybe_prep_loss_output(outputs_by_scales, ['estimated', 'valid_masks'], i=i)
        img_list += maybe_prep_loss_output(outputs_by_scales, ['auto', 'reconstructions'], i=i)
        img_list += maybe_prep_loss_output(outputs_by_scales, ['auto', 'reconstruction_losses'], i=i)

        img_list += maybe_prep_loss_output(outputs_by_scales, ['hinted', 'reconstructions'], i=i)
        img_list += maybe_prep_loss_output(outputs_by_scales, ['hinted', 'reconstruction_losses'], i=i)
        img_list += maybe_prep_loss_output(outputs_by_scales, ['hinted', 'valid_masks'], i=i)

        img_list += maybe_prep_loss_output(outputs_by_scales, ['estimated', 'minimized_loss'], i=i)

        img_list += maybe_prep_loss_output(outputs_by_scales, ['reprojection_mask'], i=i)
        img_list += maybe_prep_loss_output(outputs_by_scales, ['auto_mask'], i=i)
        img_list += maybe_prep_loss_output(outputs_by_scales, ['mask'], i=i)

    return {prefix: img_list}

def maybe_prep_loss_output(outputs_by_scales, keys, caption=None, scale_idx=0, i=0):
    img_list = []
    if keys_exists(outputs_by_scales, keys):
        o = nested_get(outputs_by_scales, keys)[scale_idx]

        if caption is None:
            caption = '_'.join(keys)

        if o.dim() == 5:
            for source_idx, element in enumerate(o):
                img_list.append(prep_rgb(f'{caption}_{source_idx}', element, i=i))
        else:
            img_list.append(prep_rgb(f'{caption}', o, i=i))

    return img_list

def nested_get(dic, keys):
    for key in keys:
        dic = dic[key]
    return dic

def keys_exists(dic, keys):
    '''
        Check if keys (list of nested keys) exists in `d` (dict).
    '''
    if not isinstance(dic, dict):
        return False

    for key in keys:
        # Necessary to ensure _element is not a different indexable type (list, string, etc).
        # get() would have the same issue if that method name was implemented by a different object
        #https://stackoverflow.com/questions/43491287/elegant-way-to-check-if-a-nested-key-exists-in-a-dict
        if not isinstance(dic, dict) or key not in dic:
            return False
        dic = dic[key]

    return True


