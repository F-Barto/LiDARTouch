from networks.extractors.image.resnet import ResNetExtractor
from networks.extractors.image.dilated_resnet_C import DRN_C
from networks.common.resnet_base import build_model

from functools import partial

from networks.extractors.lidar.resnet import LiDARResNetExtractor


def select_image_extractor(extractor_name, **kwargs):

    image_extractors = {
        'resnet': partial(build_model, ResNetExtractor),
        'DRN_C': partial(build_model, DRN_C)
    }

    if extractor_name not in image_extractors: raise NotImplementedError(f'Invalid image extractor: {extractor_name}')

    return image_extractors[extractor_name](**kwargs)

def select_lidar_extractor(extractor_name, **kwargs):

    lidar_extractors = {
        'lidar-resnet': partial(build_model, LiDARResNetExtractor),
    }

    if extractor_name not in lidar_extractors: raise NotImplementedError(f'Invalid LiDAR extractor: {extractor_name}')

    return lidar_extractors[extractor_name](**kwargs)
