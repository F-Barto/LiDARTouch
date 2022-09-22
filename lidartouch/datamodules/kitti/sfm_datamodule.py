import pytorch_lightning as pl

import numpy as np
from PIL import Image
from pathlib import Path
import random
import pickle
import cv2

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from typing import Any, List, Optional, Tuple, Union

from lidartouch.datamodules.kitti.transforms import train_transforms, val_transforms, test_transforms
from lidartouch.datamodules.kitti.kitti_utils import (get_split_file_metadata, get_calib_data_for_used_capture_dates,
                                                       get_info_from_kitti_raw_path, get_available_source_views)

from lidartouch.utils.identifiers import (TARGET_VIEW, SOURCE_VIEWS, SPARSE_DEPTH, INTRINSICS,
                                          GT_DEPTH, GT_POSES, GT_TRANS_MAG)


PROJECTED_VELODYNE_DIR = 'proj_depth/velodyne'
PROJECTED_GROUNDTRUTH_DIR = 'proj_depth/groundtruth'


class KITTIRawSfMDataset(Dataset):
    def __init__(self, kitti_raw_root_dir: Union[Path, str], relative_paths: List[str],
                 image_shape: Tuple[int, int] = (384, 1280), cam2cam_calibs: Optional[dict] = None,
                 velo2cam_calibs: Optional[dict] = None, split_data: Optional[dict] = None,
                 gt_depth_root_dir: Optional[str] = None, sparse_depth_root_dir: Optional[str] = None,
                 data_transform: Any = None, source_views_indexes: Optional[List[int]] = None, random_source: int = 0,
                 gt_usage: Optional[str] = None, input_channels: int = 3, load_pose: Optional[str] = None,
                 jittering: Optional[Tuple[float, float, float, float]] = None, lidar_drop: Optional[str] = None,
                 crop_size: Tuple[int, int] = None, trim_top: bool = False, gt_sparse_depth_dilation=False):

        self.kitti_raw_root_dir = Path(kitti_raw_root_dir).expanduser()
        assert self.kitti_raw_root_dir.exists(), self.kitti_raw_root_dir

        self.split_data = split_data
        self.cam2cam_calibs = cam2cam_calibs
        self.velo2cam_calibs = velo2cam_calibs
        self.relative_paths = relative_paths
        self.data_transform = data_transform
        self.gt_sparse_depth_dilation = gt_sparse_depth_dilation

        self.sparse_depth_root_dir = sparse_depth_root_dir
        self.gt_depth_root_dir = gt_depth_root_dir

        self.source_views_indexes = source_views_indexes
        self.random_source = random_source

        # transform parameters
        self.data_transform_options = {
            'image_shape': image_shape,
            'jittering': jittering,
            #'lidar_drop': lidar_drop,
            #'crop_size': crop_size,
            #'trim_top': trim_top
        }

        if load_pose is not None:
            assert load_pose in ['imu', 'pnp']
        self.load_pose = load_pose

        self.gt_usage = gt_usage
        if gt_usage is not None:
            if not (gt_usage in ['sparse', 'gt']):
                raise ValueError(f"gt_usage should either be 'sparse' or 'gt': {gt_usage}")

            if gt_usage == 'gt':
                assert gt_depth_root_dir is not None, "By setting `gt_usage` to 'gt you required depth GT "\
                                                      "but `gt_depth_root_dir` was found as None."
                gt_depth_root_dir = Path(gt_depth_root_dir)
                assert gt_depth_root_dir.exists(), gt_depth_root_dir
                self.gt_depth_root_dir = gt_depth_root_dir
                print(f"The GT depth from LiDAR data of each frame will be loaded from {str(gt_depth_root_dir)}.")
            self.gt_usage = gt_usage

            if gt_usage == 'sparse':
                assert sparse_depth_root_dir is not None


        self.load_sparse_depth = False
        if sparse_depth_root_dir is not None:
            sparse_depth_root_dir = Path(sparse_depth_root_dir)
            assert sparse_depth_root_dir.exists(), sparse_depth_root_dir
            self.sparse_depth_root_dir = sparse_depth_root_dir
            self.load_sparse_depth = True
            print(f"The sparse depth from LiDAR data of each frame will be loaded from {str(sparse_depth_root_dir)}.")

        self.source_views_requested = (source_views_indexes is not None) and len(source_views_indexes) > 0
        if self.source_views_requested:
            if 0 in source_views_indexes:
                source_views_indexes.remove(0)
            self.source_views_indexes = sorted(source_views_indexes)

            assert self.split_data is not None

        self.random_source = random_source
        if self.random_source > 0 and self.source_views_requested:
            random_source_err_msg = f"For each sample, you asked to draw {random_source} source views but there are not " \
                                    f"enough indexes to draw from {source_views_indexes}" \
                                    f" (length = {len(source_views_indexes)})."
            assert random_source < len(self.source_views_indexes), random_source_err_msg

        assert input_channels in [1, 3]
        self.input_channels = {1: 'gray', 3: 'rgb'}[input_channels]

        self.full_res_shape = (1280, 384)  # 384, 1280 or 1242, 375 ?

    def __len__(self):
        return len(self.relative_paths)

    def read_npz_depth(self, file):
        """Reads a .npz depth map from https://github.com/TRI-ML/packnet-sfm/."""
        try:
            depth = np.load(file)['velodyne_depth'].astype(np.float32)
        except:
            print(file)
            raise
        return np.expand_dims(depth, axis=2)

    def read_png_depth(self, file, resize=None):
        """Reads a .png depth map anbd optionally resize it."""
        depth_png = Image.open(file)
        if resize is not None:
            depth_png = depth_png.resize(resize, Image.NEAREST)

        depth_png = np.array(depth_png, dtype=int)
        # check that data encoded in 16bits not in 8bits
        assert (depth_png.max() > 255), 'Wrong .png depth file'
        depth = depth_png.astype(np.float) / 256.
        return np.expand_dims(depth, axis=-1)

    def __getitem__(self, idx):

        sample = {'idx': idx}

        target_view_path = self.kitti_raw_root_dir / self.relative_paths[idx]

        # e.g., img_path == path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png
        capture_date, sequence, camera, filename, frame_idx = get_info_from_kitti_raw_path(target_view_path)

        ########################### target view ###########################
        target_img = Image.open(target_view_path)
        if self.input_channels == 'gray':
            target_img = target_img.convert('L')
        sample[TARGET_VIEW] = target_img

        ########################### source views ###########################
        if self.random_source > 0 and self.source_views_requested:
            source_views_indexes = random.sample(range(len(self.source_views_indexes)), self.random_source)
        else:
            source_views_indexes = self.source_views_indexes

        if self.source_views_requested:
            source_views_paths, _ = get_available_source_views(self.split_data, target_view_path, source_views_indexes)
            source_views_imgs = [Image.open(self.kitti_raw_root_dir /path) for path in source_views_paths]
            if self.input_channels == 'gray':
                source_views_imgs = [source_views_img.convert('L') for source_views_img in source_views_imgs]
            sample[SOURCE_VIEWS] = source_views_imgs

        ########################### intrinsics ###########################
        K = self.cam2cam_calibs[capture_date][camera.replace('image', 'K')] # e.g., 'image_02' -> K_02
        sample[INTRINSICS] = K[:3,:3] # get intrinsics matrix to 3x3 in case it is homogeneous (4x4)

        ########################### filename (kind of internal file id) ###########################
        sequence_idx = sequence[17:21] # 2011_09_26_drive_0048_sync -> 0048
        sample['filename'] = f"{capture_date}_{sequence_idx}_c{camera[-1]}_{frame_idx:010d}"

        ########################### pose data ###########################
        if self.load_pose:
            if not self.source_views_requested:
                raise ValueError("You asked to load pose but no temporal context is defined. "
                                 "Please set `source_views_requested`.")

            if self.load_pose == 'imu':
                poses_key = 'imu_pose'
                translation_magnitudes_key = 'imu_translation_magnitude'
            elif self.load_pose == 'pnp':
                poses_key = 'pnp_pose'
                translation_magnitudes_key = 'pnp_translation_magnitude'
            else:
                raise NotImplementedError

            poses = []
            translation_magnitudes = []
            for list_idx, source_view_index in enumerate(source_views_indexes):
                source_view_data = self.split_data[capture_date][sequence][camera][frame_idx][source_view_index]
                poses.append(source_view_data[poses_key])
                translation_magnitudes.append(source_view_data[translation_magnitudes_key])

                if source_view_data[translation_magnitudes_key] == 0.:
                    sample[SOURCE_VIEWS][list_idx] = target_img

            # convert to numpy array
            sample[GT_POSES] = np.stack(poses, axis=0)
            sample[GT_TRANS_MAG] = np.stack(translation_magnitudes, axis=0)

        ########################### sparse depth ###########################
        # assumes the depth files are stored in the same format as KITTI_raw:
        # depth_root_dir/2011_09_26/2011_09_26_drive_0048_sync/proj_depth/velodyne/image_02/0000000085.npz
        sparse_depth_path = Path(self.sparse_depth_root_dir) / capture_date \
                            / f"{capture_date}_drive_{sequence_idx}_sync" / PROJECTED_VELODYNE_DIR / camera \
                            / f"{frame_idx:010d}.npz"
        if self.load_sparse_depth:
            depth = self.read_npz_depth(str(sparse_depth_path))
            sample[SPARSE_DEPTH] = depth

        ########################### gt depth ###########################
        if self.gt_usage is not None:
            if self.gt_usage == 'sparse':
                # in case we want to use the input sparse depth as ground-truth to train
                gt_depth = sample[SPARSE_DEPTH]
                if self.gt_sparse_depth_dilation:
                    kernel = np.ones((10, 10), np.uint8)
                    gt_depth = cv2.dilate(gt_depth, kernel, iterations=2)
                sample[GT_DEPTH] = gt_depth

            if self.gt_usage == 'gt':
                path_suffix = f"{capture_date}_drive_{sequence_idx}_sync/{PROJECTED_GROUNDTRUTH_DIR}/" \
                              f"{camera}/{frame_idx:010d}.png"

                # We use GT depth from the KITTI Depth Completion set and is available for both its train and val split
                projected_lidar_path = Path(self.gt_depth_root_dir) / 'val' / path_suffix

                if not projected_lidar_path.exists():
                    projected_lidar_path = Path(self.gt_depth_root_dir) / 'train' / path_suffix

                # Somehow there two image size in KITTI, so we systematically resize at biggest one
                projected_lidar = self.read_png_depth(projected_lidar_path, resize=self.full_res_shape)
                sample[GT_DEPTH] = projected_lidar

        if self.data_transform is not None:
            sample = self.data_transform(sample, **self.data_transform_options)

        return sample


class KITTIRawSfMDataModule(pl.LightningDataModule):
    """
    doc string for all args 0.0
    """

    def __init__(self, kitti_raw_root_dir: str, split_file: str, image_shape: Tuple[int, int] = (384, 1280),
                 val_split_file: Optional[str] = None, split_data: Optional[str] = None, input_channels: int = 3,
                 batch_size: int = 4, pin_memory: bool = True, num_workers: int = 10,
                 gt_depth_root_dir: Optional[str] = None, sparse_depth_root_dir: Optional[str] = None,
                 source_views_indexes: Optional[List[int]] = None, random_source: int = 0,
                 train_gt_usage: Optional[str] = None, val_gt_usage: str = 'gt', load_pose: Optional[str] = None,
                 jittering: Optional[Tuple[float, float, float, float]] = None, lidar_drop: Optional[str] = None,
                 crop_size: Tuple[int, int] = None, trim_top: bool = False, gt_sparse_depth_dilation: bool = False):

        super().__init__()
        self.save_hyperparameters()

        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.source_views_indexes = source_views_indexes

        # transforms

        self.common_kwargs = {
            'image_shape': image_shape,
            'sparse_depth_root_dir': sparse_depth_root_dir,
            'input_channels': input_channels,
            'gt_depth_root_dir': gt_depth_root_dir
        }
        self.train_kwargs = {
            'source_views_indexes': source_views_indexes,
            'random_source': random_source,
            'gt_usage': train_gt_usage,
            'load_pose': load_pose,
            'jittering': jittering,
            'lidar_drop': lidar_drop,
            'crop_size': crop_size,
            'trim_top': trim_top,
            'gt_sparse_depth_dilation': gt_sparse_depth_dilation,
        }
        self.eval_kwargs = {'gt_usage': val_gt_usage}

        self.split_data_path = None
        if split_data is not None:
            self.split_data_path = Path(split_data).expanduser()
            assert self.split_data_path.exists(), self.split_data_path

        self.kitti_raw_root_dir = Path(kitti_raw_root_dir).expanduser()
        assert self.kitti_raw_root_dir.exists(), self.kitti_raw_root_dir

        self.split_file = Path(split_file).expanduser()
        assert self.split_file.exists(), self.split_file

        self.val_split_file = None
        if val_split_file is not None:
            self.val_split_file = Path(val_split_file).expanduser()
            assert self.val_split_file.exists(), self.val_split_file



    def extract_valid_data_from_split_data(self, split_data: dict, paths: List[str],
                                           source_views_indexes: List[int]) -> List[str]:
        """
        Discard the samples from relative_paths that don't have avaible source views,
        where the source views are defined by source_views_indexes.

        Args:
            split_data : dict produced by `get_split_data()`
            paths : A list of paths in KITTI raw format to the data samples.
                format -> .../2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png
            source_views_indexes : The relative indexes to sample from the neighbouring views of the target view.
                The list should not contains 0 (corresponding to the target), it will be rmeoved otherwise.
                For example, source_indexes=[-1,1] will load the views at time t-1, t, t+1

        Returns:
            A curated list of paths to the data samples ensured to have source views.
        """
        if 0 in source_views_indexes:
            source_views_indexes.remove(0)

        valid_paths = []
        for path in paths:

            capture_date, sequence, camera, filename, frame_idx = get_info_from_kitti_raw_path(path)

            ignored_source_views_indexes = []
            for source_view_index in source_views_indexes:
                if split_data[capture_date][sequence][camera][frame_idx].get(source_view_index, None) is None:
                    ignored_source_views_indexes.append(source_view_index)
                    continue

            if len(ignored_source_views_indexes) == 0:
                valid_paths.append(path)

        return valid_paths

    def setup(self, stage: Optional[str] = None):
        """

        Args:
            stage: used to separate setup logic for trainer.{fit,validate,test}.
                If setup is called with stage = None, we assume all stages have been set-up.
        """

        if stage in (None, "fit"):

            if self.val_split_file is None:
                raise ValueError("Please define a split file for validation loop during training (val_split_file).")



            relative_paths, capture_dates, sequences, cameras = get_split_file_metadata(self.split_file)
            cam2cam, velo2cam = get_calib_data_for_used_capture_dates(capture_dates, self.kitti_raw_root_dir)
            split_data = None
            if self.split_data_path is not None:
                with open(self.split_data_path, 'rb') as f:
                    split_data = pickle.load(f)
                relative_paths = self.extract_valid_data_from_split_data(split_data, relative_paths,
                                                                         self.source_views_indexes)
            self.train_dataset = KITTIRawSfMDataset(self.kitti_raw_root_dir, relative_paths, cam2cam_calibs=cam2cam,
                                                    velo2cam_calibs=velo2cam, split_data=split_data,
                                                    data_transform=train_transforms,
                                                    **self.common_kwargs, **self.train_kwargs)


            relative_paths, capture_dates, sequences, cameras = get_split_file_metadata(self.val_split_file)
            cam2cam, velo2cam = get_calib_data_for_used_capture_dates(capture_dates, self.kitti_raw_root_dir)
            self.val_dataset = KITTIRawSfMDataset(self.kitti_raw_root_dir, relative_paths, cam2cam_calibs=cam2cam,
                                                  velo2cam_calibs=velo2cam, data_transform=val_transforms,
                                                  **self.common_kwargs, **self.eval_kwargs)

        if stage in (None, "test"):
            relative_paths, capture_dates, sequences, cameras = get_split_file_metadata(self.split_file)
            cam2cam, velo2cam = get_calib_data_for_used_capture_dates(capture_dates, self.kitti_raw_root_dir)
            self.test_dataset = KITTIRawSfMDataset(self.kitti_raw_root_dir, relative_paths, cam2cam_calibs=cam2cam,
                                                   velo2cam_calibs=velo2cam,data_transform=test_transforms,
                                                   **self.common_kwargs, **self.eval_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory,
                          num_workers=self.num_workers)


