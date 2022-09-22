import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from  torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from PIL import Image

########################################################################################################################

from lidartouch.utils.identifiers import TARGET_VIEW, SOURCE_VIEWS, SPARSE_DEPTH, DEPTH_KEYS, INTRINSICS


class LidarDropBlock2D:
    r"""
    Randomly zeroes 2D spatial blocks of the input image.

    Inspired by
        _DropBlock: A regularization method for convolutional networks
        (https://arxiv.org/abs/1810.12890)

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(H, W)`
        - Output: `(H, W)`
    """

    def __init__(self, drop_prob, block_size, height, width):
        super().__init__()

        self.block_size = block_size

        self.pad = block_size // 4

        self.kernel_size = (block_size, block_size)

        valid_area = (height - block_size // 2 + 1) * (width - block_size // 2 + 1)
        self.gamma = (drop_prob / block_size ** 2) * ((height * width) / valid_area)

        self.valid_region = (height - block_size // 2, width - block_size // 2)

    def drop(self, x, return_mask=False, return_inverse=False):

        assert x.ndim == 2, \
            "Expected input with 2 dimensions (height, width, )"

        if self.gamma == 0.:
            return x
        else:

            # bernouli sampling of block centers on valid region
            mask_valid = np.random.rand(*self.valid_region)
            mask_valid = mask_valid < self.gamma
            mask = np.pad(mask_valid, self.pad, constant_values=0)

            block_mask = self.compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            outputs = [out]

            if return_inverse:
                outputs.append(x * (1 - block_mask))

            if return_mask:
                outputs.append(block_mask)

            return tuple(outputs)

    def compute_block_mask(self, mask):

        sampled_indexes = np.array(np.nonzero(mask)).T

        block_mask = np.ones(mask.shape)

        for sampled_index in sampled_indexes:
            y, x = sampled_index

            upper_left_y = y + self.block_size // 2
            upper_left_x = x - self.block_size // 2

            bottom_left_y = y - self.block_size // 2
            bottom_left_x = x + self.block_size // 2

            cv2.rectangle(block_mask,  # source image
                          (upper_left_x, upper_left_y),  # upper left corner vertex
                          (bottom_left_x, bottom_left_y),  # lower right corner vertex
                          0,  # color
                          thickness=-1,  # filling
                          )

        return block_mask

    def run_stats(self, n=1000):
        from pprint import pprint

        cum = {}

        for _ in range(n):
            mask_valid = np.random.rand(*self.valid_region)
            mask_valid = mask_valid < self.gamma

            nb_sampled_pts = mask_valid.sum()

            cum[nb_sampled_pts] = cum.get(nb_sampled_pts, 0) + 1

        pprint(cum)
        pprint({k: v / n * 100 for k, v in cum.items()})


class LidarDropCircle2D:
    r"""
    Randomly zeroes 2D spatial blocks of the input image.

    Inspired by
        _DropBlock: A regularization method for convolutional networks
        (https://arxiv.org/abs/1810.12890)

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(H, W)`
        - Output: `(H, W)`
    """

    def __init__(self, drop_prob, block_size, height, width):
        super().__init__()

        self.block_size = block_size

        self.pad = self.block_size // 4

        valid_area = (height - block_size // 2 + 1) * (width - block_size // 2 + 1)

        circle_area = np.pi * (block_size // 2) ** 2
        self.gamma = (drop_prob / circle_area) * ((height * width) / valid_area)

        self.valid_region = (height - block_size // 2, width - block_size // 2)
        
    def drop(self, x, return_mask=False, return_inverse=False):

        assert x.ndim == 2, \
            "Expected input with 2 dimensions (height, width, )"

        if self.gamma == 0.:
            return x
        else:

            # bernouli sampling of block centers on valid region
            mask_valid = np.random.rand(*self.valid_region)
            mask_valid = mask_valid < self.gamma
            mask = np.pad(mask_valid, self.pad, constant_values=0)

            block_mask = self.compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            outputs = [out]

            if return_inverse:
                outputs.append(x * (1 - block_mask))

            if return_mask:
                outputs.append(block_mask)

            return tuple(outputs)

    def compute_block_mask(self, mask):

        sampled_indexes = np.array(np.nonzero(mask)).T

        block_mask = np.ones(mask.shape)

        radius = self.block_size // 2

        for sampled_index in sampled_indexes:
            y, x = sampled_index

            cv2.circle(
                block_mask,  # source image
                (x, y),  # center
                radius,  # radius
                0,  # color or border
                thickness=-1,  # line thickness
            )

        return block_mask

    def run_stats(self, n=1000):
        from pprint import pprint

        cum = {}

        for _ in range(n):
            mask_valid = np.random.rand(*self.valid_region)
            mask_valid = mask_valid < self.gamma

            nb_sampled_pts = mask_valid.sum()

            cum[nb_sampled_pts] = cum.get(nb_sampled_pts, 0) + 1

        pprint(cum)
        pprint({k: v / n * 100 for k, v in cum.items()})


class LidarDropUniform2D:
    r"""
    Randomly zeroes 2D spatial blocks of the input image.

    Inspired by
        _DropBlock: A regularization method for convolutional networks
        (https://arxiv.org/abs/1810.12890)

    Args:
        drop_prob (float): probability of an element to be dropped.

    Shape:
        - Input: `(H, W)`
        - Output: `(H, W)`
    """

    def __init__(self, drop_prob):
        super().__init__()

        self.drop_prob = drop_prob

    def drop(self, x, return_inverse=False):

        assert x.ndim == 2, \
            "Expected input with 2 dimensions (height, width, )"

        if self.drop_prob == 0.:
            return x
        else:

            valid_points = x > 0
            dropped_lidar_points = (np.random.rand(valid_points.sum()) < self.drop_prob)

            keeped_lidar = np.zeros(x.shape)
            dropped_lidar = np.zeros(x.shape)

            dropped_lidar[valid_points] = x[valid_points] * dropped_lidar_points
            keeped_lidar[valid_points] = x[valid_points] * ~dropped_lidar_points

            outputs = [keeped_lidar]

            if return_inverse:
                outputs.append(dropped_lidar)

            return tuple(outputs)

def resize_image(image, shape, interpolation=InterpolationMode.BICUBIC):
    """
    Resizes input image.
    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode
    Returns
    -------
    image : Image.PIL
        Resized image
    """
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)

def resize_depth(depth, shape):
    """
    Resizes depth map.
    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape
    Returns
    -------
    depth : np.array [H,W]
        Resized depth map
    """
    depth = cv2.resize(depth, dsize=shape[::-1],
                       interpolation=cv2.INTER_NEAREST)
    return depth

def random_crop_sample_and_intrinsics(sample, crop_size, trim_top=True):
    (orig_w, orig_h) = sample[TARGET_VIEW].size
    (out_h, out_w) = crop_size

    # trim the 100 top pixels (no LiDAR in this zone)
    top_margin = 0
    if trim_top:
        top_margin = 100

    assert out_w <= orig_w and out_h <= (orig_h - top_margin), \
        "crop size is larger than the input size"

    h_start = random.randint(top_margin, orig_h - out_h)
    w_start = random.randint(0, orig_w - out_w)

    sample[TARGET_VIEW] = TF.crop(sample[TARGET_VIEW],h_start, w_start, out_h, out_w)

    keys = DEPTH_KEYS
    for key in keys:
        if sample.get(key) is not None:
            tmp_sample = Image.fromarray(sample[key].astype('float32'), mode='F')
            tmp_sample = TF.crop(tmp_sample, h_start, w_start, out_h, out_w)
            sample[key] = np.array(tmp_sample).astype('float32')

    # adjusting cx and cy
    key = INTRINSICS
    intrinsics = np.copy(sample[key])
    intrinsics[0][2] -= w_start
    intrinsics[1][2] -= h_start
    sample[key] = intrinsics



def resize_sample_image_and_intrinsics(sample, shape, image_interpolation=InterpolationMode.BICUBIC):
    """
    Resizes the image and intrinsics of a sample
    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode
    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and corresponding intrinsics
    image_transform = transforms.Resize(shape, interpolation=image_interpolation)
    (orig_w, orig_h) = sample[TARGET_VIEW].size
    (out_h, out_w) = shape

    # Scale intrinsics
    key = INTRINSICS
    intrinsics = np.copy(sample[key])
    x_scale = out_w / orig_w
    y_scale = out_h / orig_h
    intrinsics[0] *= x_scale
    intrinsics[1] *= y_scale
    sample[key] = intrinsics

    # Scale target image
    key = TARGET_VIEW
    sample[key] = image_transform(sample[key])

    # Scale source views images
    key = SOURCE_VIEWS
    if sample.get(key) is not None:
        sample[key] = [image_transform(source_view) for source_view in sample[key]]


    # Return resized sample
    return sample

def resize_sample(sample, shape, image_interpolation=InterpolationMode.BICUBIC):
    """
    Resizes a sample, including image, intrinsics and depth maps.
    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode
    Returns
    -------
    sample : dict
        Resized sample
    """

    # Resize image and intrinsics
    sample = resize_sample_image_and_intrinsics(sample, shape, image_interpolation)

    # Resize depth maps
    for key in DEPTH_KEYS:
        if sample.get(key) is not None:
            sample[key] = resize_depth(sample[key], shape)

    # Return resized sample
    return sample


def to_tensor(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)

def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.
    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to
    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """

    # Convert single items
    for key in ([TARGET_VIEW, TARGET_VIEW + '_original', SPARSE_DEPTH + '_original'] + DEPTH_KEYS):
        if sample.get(key) is not None:
            sample[key] = to_tensor(sample[key], tensor_type)

    # Convert lists
    for key in [SOURCE_VIEWS, SOURCE_VIEWS + '_original']:
        if sample.get(key) is not None:
            sample[key] = [to_tensor(source_view, tensor_type) for source_view in sample[key]]

    # Return converted sample
    return sample


def duplicate_sample(sample):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.
    Parameters
    ----------
    sample : dict
        Input sample
    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """

    # Duplicate target image
    key = TARGET_VIEW
    sample[f'{key}_original'] = sample[key].copy()

    # Duplicate source view images
    key = SOURCE_VIEWS
    if sample.get(key) is not None:
        sample[f'{key}_original'] = [k.copy() for k in sample[key]]

    # Duplicate (?) sparse projected lidar image
    key = SPARSE_DEPTH
    if sample.get(key) is not None:
        sample[f'{key}_original'] = sample[key].copy()

    # Return duplicated sample
    return sample

def colorjitter_sample(sample, parameters, prob=1.0):
    """
    Jitters input images as data augmentation.
    Parameters
    ----------
    sample : dict
        Input sample
    parameters : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    prob : float
        Jittering probability
    Returns
    -------
    sample : dict
        Jittered sample
    """
    if random.random() < prob:
        # Prepare transformation
        brightness, contrast, saturation, hue = parameters
        augment_image = transforms.ColorJitter(
            brightness=[max(0, 1 - brightness), 1 + brightness],
            contrast=[max(0, 1 - contrast), 1 + contrast],
            saturation=[max(0, 1 - saturation), 1 + saturation],
            hue=[-hue, hue])

        # Jitter target image
        key = TARGET_VIEW
        sample[key] = augment_image(sample[key])

        # Jitter source views images
        key = SOURCE_VIEWS
        if sample.get(key) is not None:
            sample[key] = [augment_image(source_view) for source_view in sample[key]]

    # Return jittered (?) sample
    return sample


def sparse_lidar_drop(sample,  height, width, drop_scheme, drop_prob, supervise_on_dropped=False, drop_size=50):
    """
    Jitters input images as data augmentation.
    Parameters
    ----------
    sample : dict
        Input sample
    parameters : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    prob : float
        Jittering probability
    Returns
    -------
    sample : dict
        Jittered sample
    """

    assert drop_scheme in ['circle', 'block', 'full', 'uniform']

    key = SPARSE_DEPTH

    if sample.get(key, None) is None or drop_prob == 0.:
        return sample

    if drop_scheme == 'full':
        if np.random.rand(1) < drop_prob:
            sample[key] = np.zeros(sample[key].shape)
        return sample

    if drop_scheme == 'block':
        lidar_drop = LidarDropBlock2D(drop_prob, drop_size, height, width)
    elif drop_scheme == 'circle':
        lidar_drop = LidarDropCircle2D(drop_prob, drop_size, height, width)
    else: # drop_scheme == 'uniform'
        lidar_drop = LidarDropUniform2D(drop_prob)

    sample_lidar = sample[key]

    # sparse projected lidar is usually given as shape (H, W, 1)
    if sample_lidar.ndim == 3:
        sample_lidar = sample_lidar[:,:,0]

    lidar_outputs = lidar_drop.drop(sample_lidar, return_inverse=supervise_on_dropped)

    if supervise_on_dropped:
        # use the dropped part as supervision for exclusive supervision
        sample[key+'_original'] = np.expand_dims(lidar_outputs[1], axis=2)

    sample[key] = np.expand_dims(lidar_outputs[0], axis=2)

    return sample

###################################################

def train_transforms(sample, image_shape, jittering=None, lidar_drop=None, random_crop=None):
    """
    Training data augmentation transformations
    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    Returns
    -------
    sample : dict
        Augmented sample
    """
    image_shape = tuple(image_shape)
    sample = resize_sample(sample, image_shape)

    if random_crop is not None and len(random_crop) > 0:
        random_crop_sample_and_intrinsics(sample, **random_crop)

    sample = duplicate_sample(sample)

    if jittering is not None and len(jittering) > 0:
        jittering = tuple(jittering)
        sample = colorjitter_sample(sample, jittering)

    if lidar_drop is not None and len(lidar_drop) > 0:
        sample = sparse_lidar_drop(sample, *image_shape, *lidar_drop)

    sample = to_tensor_sample(sample)
    return sample

def val_transforms(sample, image_shape, **kwargs):
    """
    Validation data augmentation transformations
    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    Returns
    -------
    sample : dict
        Augmented sample
    """
    image_shape = tuple(image_shape)

    sample = resize_sample_image_and_intrinsics(sample, image_shape)

    # Resize sparse depth maps
    if sample.get(SPARSE_DEPTH) is not None:
        sample[SPARSE_DEPTH] = resize_depth(sample[SPARSE_DEPTH], image_shape)

    # GT depth is not resized

    sample = to_tensor_sample(sample)
    return sample

def test_transforms(sample, image_shape, **kwargs):
    """
    Test data augmentation transformations
    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    Returns
    -------
    sample : dict
        Augmented sample
    """
    image_shape = tuple(image_shape)

    sample = resize_sample_image_and_intrinsics(sample, image_shape)

    # Resize sparse depth maps
    if sample.get(SPARSE_DEPTH) is not None:
        sample[SPARSE_DEPTH] = resize_depth(sample[SPARSE_DEPTH], image_shape)

    # GT depth is not resized

    sample = to_tensor_sample(sample)
    return sample