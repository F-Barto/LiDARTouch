from lidartouch.networks.layers.fusion.elem_wise_fusions import ElemWiseMultFusion, ElemWiseSumFusion
from lidartouch.networks.layers.fusion.concat import ConcatFusion

import numpy as np

def select_fusion_module(fusion_name):

    fusions = {
        'mult': ElemWiseMultFusion,
        'sum': ElemWiseSumFusion,
        'concat': ConcatFusion,
    }

    if fusion_name not in fusions: raise NotImplementedError(f'Invalid image extractor: {fusion_name}')

    return fusions[fusion_name]


def get_ch_post_fusion(fusion_name, lidar_ch_enc, image_ch_enc):
    lidar_ch_enc = np.array(lidar_ch_enc)
    image_ch_enc = np.array(image_ch_enc)

    max_between_twos = np.stack([lidar_ch_enc, image_ch_enc]).max(axis=0)

    fusions = {
        'mult': max_between_twos,
        'sum': max_between_twos,
        'concat': lidar_ch_enc + image_ch_enc,
        'spatial': max_between_twos,
        'project_squeeze_fuse': max_between_twos,
    }

    if fusion_name not in fusions: raise NotImplementedError(f'Invalid image extractor: {fusion_name}')

    return list(fusions[fusion_name])
