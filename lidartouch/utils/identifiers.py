

#########################################
##### KITTISFMDATAMODULE BATCH KEYS #####
#########################################

TARGET_VIEW = 'target_view'
SOURCE_VIEWS = 'source_views'
SPARSE_DEPTH = 'sparse_depth'
GT_DEPTH = 'gt_depth'
DEPTH_KEYS = [GT_DEPTH, SPARSE_DEPTH]
INTRINSICS = 'intrinsics'
GT_POSES = 'gt_poses'
GT_TRANS_MAG = 'gt_translation_magnitudes'


###############################
##### DEPTH_NET SIGNATURE #####
###############################

IMAGE = 'camera_image'
SPARSE_DEPTH = "sparse_depth"
# CAMERA_MATRIX = 'camera_matrix'

depthnet_to_batch_keys = {
    IMAGE: TARGET_VIEW,
    SPARSE_DEPTH: SPARSE_DEPTH
}

#######################
##### OUTPUT KEYS #####
#######################

INV_DEPTHS = "inv_depths"



