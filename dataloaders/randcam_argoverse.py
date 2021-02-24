"""
This module implements a Dataloader suitable for the self-supervisied training of a monocular depth estimation method on
the argoverse Tracking dataset (please cite their work and our work if you use it).

assumes that the argoverse API is installed

In this module, the docstring follows the NumPy/SciPy formatting rules.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from random import randrange
import skimage.io
import pickle

import cv2

from utils.pose_estimator import get_pose_pnp

from pytorch_lightning import _logger as terminal_logger

# parametric continuous conv pre-requisites
from sklearn.neighbors import KDTree
from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t
from argoverse.utils.calibration import point_cloud_to_homogeneous
from argoverse.utils.ply_loader import load_ply






VEHICLE_CALIBRATION_INFO_FILENAME = 'vehicle_calibration_info.json'



def determine_valid_cam_coords(uv, uv_cam, img_width, img_height):
    """
    adatpted from argoverse-api

    Given a set of coordinates in the image plane and corresponding points
    in the camera coordinate reference frame, determine those points
    that have a valid projection into the image. 3d points with valid
    projections have x coordinates in the range [0,img_width-1], y-coordinates
    in the range [0,img_height-1], and a positive z-coordinate (lying in
    front of the camera frustum).
    Args:
       uv: Numpy array of shape (N,2)
       uv_cam: Numpy array of shape (3, N) (or 4xN if homogeneous)
       img_width: width of image on whish the projection is done
       img_height: height of image on whish the projection is done
    Returns:
       Numpy array of shape (N,) with dtype bool
    """
    x_valid = np.logical_and(0 <= uv[:, 0], uv[:, 0] < img_width)
    y_valid = np.logical_and(0 <= uv[:, 1], uv[:, 1] < img_height)
    z_valid = uv_cam[2, :] > 0
    valid_pts_bool = np.logical_and(np.logical_and(x_valid, y_valid), z_valid)
    return valid_pts_bool

def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K


def compute_multi_scale_continuous_conv_prerequisites(camera_config, scales, lidar_pts, lidar_timestamp, cam_timestamp,
                                                      dataset_dir, log_id, K=3):
    """
    A continuous convolution as defined in the paper
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Deep_Parametric_Continuous_CVPR_2018_paper.pdf

    h_i = W(sum_k(MLP(x_i - x_k) * f_k))

    requires to computes:
    - the k nearest neighbors of each point in the point cloud (using a KD-Tree)
    - the difference between the point and each of its neighbors (x_i - x_k)
    - the pixel indexes on which projects each neighbors 3D points `k` to extract the feature tensor f_k
    - the pixel indexes on which projects each anchor 3D points `i` to set h_i

    this can be done offline to speed up the network training.

    Args:
        camera_config: argoverse-api CameraConfig
        scales: list of float, scaling ratio from the original image size to compute the pixel indexes at, e.g. [1/2, 1/4, 1/8, 1/16]
        lidar_pts: numpy array, 3d points coordinates from the ply file
        lidar_timestamp: int,
        cam_timestamp: int,
        dataset_dir: str,
        log_id: str,
        K: int, number of neighbors to consider

    Returns for each scale:
        nn_pixel_idxs
        pixel_idxs
        nn_diff_pts_3D
    """

    ########### motion compensated projection adapted from `project_lidar_to_img_motion_compensated` ###########

    # get transformation to bring point in egovehicle frame to city frame,
    # at the time when camera image was recorded.
    city_SE3_ego_cam_t = get_city_SE3_egovehicle_at_sensor_t(cam_timestamp, dataset_dir, log_id)

    # get transformation to bring point in egovehicle frame to city frame,
    # at the time when the LiDAR sweep was recorded.
    city_SE3_ego_lidar_t = get_city_SE3_egovehicle_at_sensor_t(lidar_timestamp, dataset_dir, log_id)

    if city_SE3_ego_cam_t is None or city_SE3_ego_lidar_t is None:
        return None, None, None

    pts_h_lidar_time = point_cloud_to_homogeneous(lidar_pts).T

    # convert back from homogeneous
    pts_h_lidar_time = pts_h_lidar_time.T[:, :3]
    ego_cam_t_SE3_ego_lidar_t = city_SE3_ego_cam_t.inverse().right_multiply_with_se3(city_SE3_ego_lidar_t)
    pts_h_cam_time = ego_cam_t_SE3_ego_lidar_t.transform_point_cloud(pts_h_lidar_time)
    pts_h_cam_time = point_cloud_to_homogeneous(pts_h_cam_time).T

    uv_cam = camera_config.extrinsic.dot(pts_h_cam_time)  # 3D lidar points homogeneous coordinates in camera reference frame

    ######## neareast neigbors computation ########

    by_scale_nn_pixel_idxs = []
    by_scale_pixel_idxs = []
    by_scale_nn_diff_pts_3d = []

    # TODO: usually the number of points visible at different differ very slightly and mostly due to rounding error
    # empirrically the number of valid 3D points increases as the image resolution is smaller
    # so we would like to compute pts_3D, nn_idxs and nn_diff_pts at the biggest scale only
    # that way the KD Tree is only computed one time

    for scale in scales:
        intrinsic = scale_intrinsics(camera_config.intrinsic.copy(), scale, scale)
        img_width, img_height = (int(camera_config.img_width * scale), int(camera_config.img_height * scale))

        uv = intrinsic.dot(uv_cam)  # 2D pixel coordiantes of projected 3D lidar points

        uv[0:2, :] /= uv[2, :]
        uv = uv.T
        uv = uv[:, :2]
        valid_pts_bool = determine_valid_cam_coords(uv, uv_cam, img_width, img_height)

        # Nx2 array, 2D pixel coordinates of VALID (i.e. in camera's frustum) projected 3D lidar points
        pixel_idxs = uv[valid_pts_bool].astype(np.long)

        # Nx3 array, valid 3D lidar points coordinates in camera reference frame
        pts_3d = uv_cam.T[valid_pts_bool][:, :3]

        # nn_idxs is a NxK array where nn_idxs[i] = i_0, ..., i_k the neighbors 3D points coords index in pts_3d
        # pts_3d[nn_idxs[i,k]] is the 3D coordinates of 3D point i's kth neigbor
        tree = KDTree(pts_3d, leaf_size=40)
        nn_idxs = tree.query(pts_3d, k=K + 1, return_distance=False)
        nn_idxs = nn_idxs[:, 1:]  # dropping first column as nearest neighboor includes itself

        # nn_pixel_idxs is a NxKx2 array where nn_pixel_idxs[i] = [[x_0, y_0], ..., [x_k, y_k]]
        # the 2D pixel coordinates of the projected neighbors points
        nnks_pixel_idxs = [np.expand_dims(pixel_idxs[nn_idxs[:, k]], axis=1) for k in range(K)]
        nn_pixel_idxs = np.concatenate(nnks_pixel_idxs, axis=1)

        # nn_diff_pts_3d is a NxKx3 array where nn_diff_pts_3d[i,k] = x_i - x_k
        # the 3D coordinates of the difference
        nn_diff_pts_3d = [np.expand_dims(pts_3d - pts_3d[nn_idxs[:, k]], axis=1) for k in range(K)]
        nn_diff_pts_3d = np.concatenate(nn_diff_pts_3d, axis=1)

        by_scale_nn_pixel_idxs.append(nn_pixel_idxs)
        by_scale_pixel_idxs.append(pixel_idxs)
        # /!\ very import to cast to float32 for pytorch, unpredictable CUDA errors could occurs later in the model
        by_scale_nn_diff_pts_3d.append(nn_diff_pts_3d.astype(np.float32))

    return by_scale_nn_pixel_idxs, by_scale_pixel_idxs, by_scale_nn_diff_pts_3d




class RandCamSequentialArgoverseLoader(Dataset):
    """
    argoverse Tracking Dataloader that loads:
        - the image at time t (the target view)
        - the neighbouring images, e.g., at time t-1 and t+1 (the source views)
        - the depth image from the LiDAR 3D data
        - the pose

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(self, argoverse_tracking_root_dir,  gt_depth_root_dir=None,
                 sparse_depth_root_dir=None, data_transform=None, data_transform_options=None,
                 load_pose=False, split_file=None, input_channels=3, fix_cam_idx=None, nn_precompute=False,
                 nn_scales=[0.4, 0.4 / 2, 0.4 / 4 , 0.4 / 8], use_pnp=False):

        """
        Parameters
        ----------
        argoverse_tracking_root_dir : str
            The path to the root of the argoverse tracking dataset,
            e.g., /home/clear/fbartocc/data/ARGOVERSE/argoverse-tracking
        gt_depth_root_dir : str
            The path to where the computed depth maps are stored.
            If not None, the depth from LiDAR data of each frame will be returned.
        data_transform :
            Transform applied to each data sample before returning it.
        source_views_indexes : list of int
            The relative indexes to sample from the neighbouring views of the target view.
            It is expected that the list is in ascending order and does not contains 0 (corresponding to the target).
            For example, source_indexes=[-1,1] will load the views at time t-1, t, t+1
        load_pose : bool
            If True, the pose of each frame will be returned.
        """
        super().__init__()

        self.load_pose = load_pose
        self.use_pnp = use_pnp

        self.nn_precompute = nn_precompute
        assert len(nn_scales) > 0
        self.nn_scales = nn_scales

        argoverse_tracking_root_dir = Path(argoverse_tracking_root_dir).expanduser()
        assert argoverse_tracking_root_dir.exists(), argoverse_tracking_root_dir
        self.argoverse_tracking_root_dir = argoverse_tracking_root_dir

        assert input_channels in [1,3], input_channels
        self.input_channels = {1: 'gray', 3: 'rgb'}[input_channels]

        self.data_transform = data_transform
        self.data_transform_options = data_transform_options

        if data_transform is not None:
            assert data_transform_options is not None

        if gt_depth_root_dir is not None:
            gt_depth_root_dir = Path(gt_depth_root_dir)
            assert gt_depth_root_dir.exists(), gt_depth_root_dir
            self.gt_depth_root_dir = gt_depth_root_dir
            terminal_logger.info(f"The GT depth from LiDAR data of each frame will be loaded from {str(gt_depth_root_dir)}.")

        self.load_sparse_depth = False
        if sparse_depth_root_dir is not None:
            sparse_depth_root_dir = Path(sparse_depth_root_dir)
            assert sparse_depth_root_dir.exists(), sparse_depth_root_dir
            self.sparse_depth_root_dir = sparse_depth_root_dir
            self.load_sparse_depth = True
            terminal_logger.info(
                f"The sparse depth from LiDAR data of each frame will be loaded from {str(sparse_depth_root_dir)}.")

        assert split_file is not None
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
        source_views_indexes = split_data['source_views_indexes']
        self.split_name = split_data['split_name']
        self.camera_list = split_data['camera_list']
        self.camera_configs = split_data['camera_configs']
        self.samples_paths = split_data['samples_paths']
        self.translation_magnitudes = split_data.get('translation_magnitudes', None)

        self.fix_cam_idx = None
        if fix_cam_idx is not None:
            assert fix_cam_idx < len(self.camera_list), f"{len(self.camera_list)} cameras | {self.camera_list}"
            self.fix_cam_idx = fix_cam_idx


        assert self.split_name in ['train', 'val', 'test']

        self.source_views_requested = source_views_indexes is not None and len(source_views_indexes) > 0

        # filter out samples where gt depth is not available for val and test
        if self.split_name in ['val', 'test']:
            terminal_logger.info("Filtering out samples which don't have ground truth depth")
            self.samples_paths = [sample for sample in self.samples_paths if self.has_gt_depth(sample)]

        terminal_logger.info(f'Dataset for split {self.split_name} ready.\n\n' + '-'*90 + '\n\n')

    def has_gt_depth(self, sample):
        lidar_path = sample[0]
        # we only need to test for one camera, as the only case where gt is not available is when we are at log
        # boundaries so we can't accumulate lidar sweeps. In that case gt is not available for all cams.
        camera_name = self.camera_list[0]
        gt_depth_filepath = self.get_projected_lidar_path(camera_name, lidar_path, self.gt_depth_root_dir)

        return Path(gt_depth_filepath).exists()


    def get_projected_lidar_path(self, camera_name, lidar_path, depth_base_dir):
        lidar_path = Path(lidar_path)
        split = lidar_path.parents[2].stem
        log = lidar_path.parents[1].stem
        return str(depth_base_dir / split / log / camera_name / (lidar_path.stem + '.npz'))

    def read_tiff_depth(self, file_path):
        depth = skimage.io.imread(file_path).astype(np.float32)
        return np.expand_dims(depth, axis=2)

    def read_npz_depth(self, file):
        depth = np.load(file)['arr_0'].astype(np.float32)
        return np.expand_dims(depth, axis=2)

    def load_img(self, file_path):
        img = Image.open(file_path)
        if self.input_channels == 'gray':
            return img.convert('L')
        return img

    def __len__ (self):
        return len(self.samples_paths)


    def __getitem__(self, idx):

        if self.fix_cam_idx is not None:
            cam_idx = self.fix_cam_idx
        else:
            if self.split_name == 'train':
                cam_idx = randrange(len(self.camera_list))
            else:
                cam_idx = idx % len(self.camera_list)

        camera_name = self.camera_list[cam_idx]

        # self.samples_paths[idx][0] is lidar, subsequent indexes are cameras
        target_view_path = self.argoverse_tracking_root_dir / self.samples_paths[idx][1+cam_idx][0]
        target_view = self.load_img(target_view_path)

        sample = {
            'target_view': target_view,
            'idx': idx
        }

        lidar_path = self.samples_paths[idx][0]  # the .ply file

        if self.split_name in ['val', 'test']:
            projected_lidar_path = self.get_projected_lidar_path(camera_name, lidar_path, self.gt_depth_root_dir)
            projected_lidar = self.read_npz_depth(projected_lidar_path)
            sample['projected_lidar'] = projected_lidar

        if self.load_sparse_depth:
            sparse_projected_lidar_path = self.get_projected_lidar_path(camera_name, lidar_path,
                                                                        self.sparse_depth_root_dir)
            sparse_projected_lidar = self.read_npz_depth(sparse_projected_lidar_path)
            sample['sparse_projected_lidar'] = sparse_projected_lidar

        if self.source_views_requested:
            source_views_paths = [str(self.argoverse_tracking_root_dir / p)
                                  for p in self.samples_paths[idx][1 + cam_idx][1]]
            source_views = [self.load_img(p) for p in source_views_paths]
            sample['source_views'] = source_views


        if self.translation_magnitudes is not None and self.load_pose:
            sample['translation_magnitudes'] = self.translation_magnitudes[idx][cam_idx]

        split = target_view_path.parents[2].stem
        log = target_view_path.parents[1].stem
        camera_name = target_view_path.parent.stem
        image_name = Path(target_view_path).stem # format is {camera_name}_{timestamp}
        sample['filename'] = f"{split}_{log}_{image_name}"

        cam_config = self.camera_configs[log][camera_name]
        sample['intrinsics'] = cam_config.intrinsic[:3, :3]

        if self.use_pnp:
            pnp_poses = []
            for i, source_view_img in enumerate(sample['source_views']):
                success, r_vec, t_vec = get_pose_pnp(np.array(sample['target_view']),
                                                     np.array(source_view_img),
                                                     sample['sparse_projected_lidar'],
                                                     sample['intrinsics'])

                # discard if translation is too small
                success = success and np.linalg.norm(t_vec) > 0.15
                if success:
                    vec = np.concatenate([t_vec, r_vec], axis=0).flatten()
                else:
                    # return the same image and no motion when PnP fails
                    sample['source_views'][i] = sample['target_view']
                    vec = np.zeros(6)
                pnp_poses.append(vec)
            sample['poses_pnp'] = np.stack(pnp_poses, axis=0)
            print('sample[poses_pnp]: ', sample['poses_pnp'])
            print('-'*60)

        if self.nn_precompute:
            image_timestamp = image_name[len(camera_name)+1:]

            lidar_timestamp = Path(lidar_path).stem[3:]
            pc = load_ply(str(self.argoverse_tracking_root_dir / lidar_path))
            outputs = compute_multi_scale_continuous_conv_prerequisites(cam_config, self.nn_scales, pc,
                                                                        int(lidar_timestamp), int(image_timestamp),
                                                                        str(self.argoverse_tracking_root_dir / split),
                                                                        log)

            nn_pixel_idxs, pixel_idxs, nn_diff_pts_3d = outputs
            del outputs

            sample['nn_pixel_idxs'] = nn_pixel_idxs
            sample['pixel_idxs'] = pixel_idxs
            sample['nn_diff_pts_3d'] = nn_diff_pts_3d


        if self.data_transform is not None:
            self.data_transform(sample, **self.data_transform_options)

        return sample