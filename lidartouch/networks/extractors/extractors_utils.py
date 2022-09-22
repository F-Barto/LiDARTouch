from functools import partial

from lidartouch.networks.layers.resnet_base import build_network

from lidartouch.networks.extractors.lidar.resnet import LiDARResNetExtractor
from lidartouch.networks.extractors.image.resnet import ResNetExtractor

def select_image_extractor(extractor_name, **kwargs):

    image_extractors = {
        'resnet': partial(build_network, ResNetExtractor),
    }

    if extractor_name not in image_extractors: raise NotImplementedError(f'Invalid image extractor: {extractor_name}')

    return image_extractors[extractor_name](**kwargs)

def select_lidar_extractor(extractor_name, **kwargs):

    lidar_extractors = {
        'lidar-resnet': partial(build_network, LiDARResNetExtractor),
    }

    if extractor_name not in lidar_extractors: raise NotImplementedError(f'Invalid LiDAR extractor: {extractor_name}')

    return lidar_extractors[extractor_name](**kwargs)
