from lidartouch.losses.reconstruction.ssim_l1 import SSIM_L1

class Masked_SSIM_L1(SSIM_L1):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def compute_masks(self, outputs_by_source, sparse_depth_original, **kwargs):

        masks = super().compute_masks(outputs_by_source)

        masks['no_depth_data_mask'] = sparse_depth_original <= 0.  # B,1,H,W

        return masks





