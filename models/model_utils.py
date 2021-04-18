from networks.legacy.packnet.packnet import PackNet01
from networks.legacy.packnet.posenet import PoseNet
from networks.legacy.monodepth2.depth_rest_net import DepthResNet
from networks.legacy.monodepth2.pose_res_net import PoseResNet as PoseResNet_legacy
from networks.legacy.monodepth2.guided_depth_rest_net import GuidedDepthResNet
from networks.legacy.custom.guided_sparse_dilated_depth_net import GuidedSparseDepthResNet
from networks.legacy.monodepth_original.depth_res_net import DepthResNet as OriginalDepthResNet
from networks.legacy.monodepth_original.pose_res_net import PoseResNet as OriginalPoseResNet

from networks.nets.depth_nets.selfsup_sparse2dense import DepthCompletionNet
from networks.nets.depth_nets.mfmp_symmetric import MFMPDepthNet
from networks.nets.depth_nets.monodepth2 import DepthNetMonodepth2
from networks.nets.depth_nets.feature_rescaling import FeatRescaling
from networks.nets.depth_nets.disparity_rescaling import DispRescaling
from networks.nets.depth_nets.spatial_attention import SpatialAttention
from networks.nets.depth_nets.mfmp_pointnet import MFMPPointNet
from networks.nets.depth_nets.pvt_depthnet import PVTDepthNet
from networks.nets.depth_nets.drn_transformer import DRNTransformer
from networks.nets.depth_nets.infused_pvt import InfusedPVTDepthNet
from networks.nets.depth_nets.attention_bottleneck import DepthNetAttentionBottleneck
from networks.nets.depth_nets.saf_attention_bottleneck import DepthNetSAFBottleneck
from networks.nets.depth_nets.mfmp_hr_symmetric import MFMPHRDepthNet
from networks.nets.depth_nets.drn_slot import DRNSlot
from networks.nets.depth_nets.drn_early_fusion import DRNEalyFusion
from networks.nets.depth_nets.mfmp_slot import MFMPSlotDepthNet

from networks.nets.pose_nets.monodepth2 import PoseResNet


def select_depth_net(depth_net_name, depth_net_options, load_sparse_depth=False):

    sparse_depth_input_required = ['guiding', 'sparse-guiding', 'MFMP_symmetric']
    if depth_net_name in sparse_depth_input_required:
        assert load_sparse_depth, "Sparse depth signal is necessary for feature guidance."

    depth_nets = {
        'packnet': PackNet01,
        'monodepth': DepthResNet,
        'monodepth2': DepthNetMonodepth2,
        'MFMP_symmetric': MFMPDepthNet,
        'monodepth_original': OriginalDepthResNet,
        'guiding': GuidedDepthResNet,
        'sparse-guiding': GuidedSparseDepthResNet,
        'selfsup_sparse2dense': DepthCompletionNet,
        'feature_rescaling': FeatRescaling,
        'disparity_rescaling': DispRescaling,
        'spatial_attention': SpatialAttention,
        'MFMP_PointNet': MFMPPointNet,
        'pvt': PVTDepthNet,
        'drn_transformer': DRNTransformer,
        'infused_pvt': InfusedPVTDepthNet,
        'attention_bottleneck': DepthNetAttentionBottleneck,
        'saf_attention_bottleneck': DepthNetSAFBottleneck,
        'MFMP_HR_symmetric': MFMPHRDepthNet,
        'drn_slot': DRNSlot,
        'drn_early_fusion': DRNEalyFusion,
        'mfmp_slot': MFMPSlotDepthNet
    }

    if depth_net_name not in depth_nets:
        raise NotImplementedError(f"Depth network of name {depth_net_name} not implemented")

    return depth_nets[depth_net_name](**depth_net_options)

def select_pose_net(pose_net_name, pose_net_options):
    if pose_net_name == 'packnet':
        pose_net = PoseNet
    elif pose_net_name == 'monodepth':
        pose_net = PoseResNet_legacy
    elif pose_net_name == 'monodepth2':
        pose_net = PoseResNet
    elif pose_net_name == 'monodepth_original':
        pose_net = OriginalPoseResNet
    else:
        raise NotImplementedError(f"Pose network of name  {pose_net_name} not implemented")

    return pose_net(**pose_net_options)


