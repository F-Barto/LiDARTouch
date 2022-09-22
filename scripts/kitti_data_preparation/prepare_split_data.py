from pathlib import Path
import click
from tqdm import tqdm
import numpy as np
from PIL import Image
from typing import List, Union
import pickle

import imu_pose_utils
import pnp_pose_utils


from lidartouch.datamodules.kitti.kitti_utils import (aggregate_split_files_metadata, get_sequences_lengths,
                                                       get_available_source_views, get_info_from_kitti_raw_path,
                                                       get_calib_data_for_used_capture_dates)



"""
python ./prepare_split_data.py \
/home/clear/fbartocc/data/KITTI_raw/  \
/home/clear/fbartocc/working_data/KITTI/split_data.pkl \
/home/clear/fbartocc/depth_project/LiDARTouch/data_splits \
'eigen_train_files.txt,filtered_eigen_val_files.txt,filtered_eigen_test_files.txt' \
'[-3,-2,-1,1,2,3]' \
--extend \
--imu \
--pnp /home/clear/fbartocc/working_data/KITTI/MC_sparse_lidar/factor_16 \
--pnp_threshold_translation 0.15
"""

# Constants
CAM2CAM_CALIB_FILENAME = 'calib_cam_to_cam.txt'
VELO2CAM_CALIB_FILENAME = 'calib_velo_to_cam.txt'
PROJECTED_VELODYNE_DIR = 'proj_depth/velodyne'

def setup_split_data_dict(capture_dates, sequences, cameras) -> dict:
    split_data = {}
    for capture_date in capture_dates:
        split_data[capture_date] = {}

    for sequence in sequences:
        # e.g., seq_dir = 2011_09_26_drive_0001_sync
        capture_date = sequence[:10]
        split_data[capture_date][sequence] = {}

        for camera in cameras:
            split_data[capture_date][sequence][camera] = {}

    return split_data


def get_split_data(kitti_raw_root_dir: Union[Path, str], relative_paths: List[str], capture_dates: List[str],
                   sequences: List[str], cameras: List[str], source_views_indexes: List[int]) -> dict:
    """
    Generate a dict in which information between taget and source views can be stored.
    For example:
        split_data[capture_date][sequence][camera][frame_idx][source_view_index] = {
                'source_view_path': ...,
                'pose_imu': ...,
                'pose_pnp': ...,
            }

    Args:
        kitti_raw_root_dir : absolute path to the kitti raw directory
        relative_paths : list of all relative path to process in KITTI raw format
            format -> .../2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png
        capture_dates : list of all capture dates
        sequences : list of all sequences
        cameras : list of all capture
        source_views_indexes : The relative indexes to sample from the neighbouring views of the target view.
            The list should not contains 0 (corresponding to the target), it will be rmeoved otherwise.
            For example, source_indexes=[-1,1] will load the views at time t-1, t, t+1

    Returns:
        A curated list of paths to the data samples ensured to have source views.
    """

    if 0 in source_views_indexes:
        source_views_indexes.remove(0)

    kitti_raw_root_dir = Path(kitti_raw_root_dir)

    split_data = setup_split_data_dict(capture_dates, sequences, cameras)
    sequences_lengths = get_sequences_lengths(kitti_raw_root_dir, sequences, cameras)

    for rel_path in relative_paths:

        rel_path = Path(rel_path)

        capture_date, sequence, camera, filename, frame_idx = get_info_from_kitti_raw_path(rel_path)

        img_path = kitti_raw_root_dir / rel_path
        assert img_path.exists(), f'Target view not present in given dataset directory: {img_path}'

        sequence_length = sequences_lengths[sequence][camera]
        split_data[capture_date][sequence][camera][frame_idx] = {}

        ignored_source_views_indexes = []
        for source_view_index in source_views_indexes:
            source_view_frame_idx = frame_idx + source_view_index
            # Checks if source view index falls out of the sequence's bounds, if yes -> next idx
            if source_view_frame_idx < 0 or source_view_frame_idx >= sequence_length:
                ignored_source_views_indexes.append(source_view_index)
                continue

            source_view_img_path = rel_path.parent / f'{source_view_frame_idx:010d}.png'
            # File should exists, throw error if otherwise
            assert (kitti_raw_root_dir / source_view_img_path).exists(), (
                f'Source view image {source_view_frame_idx}, while not out of bound (sequence length = {sequence_length}), '
                f'is not present in given dataset directory: {source_view_img_path}'
            )

            # if we're here, then it's all good, source view is not out of bound and data is available
            split_data[capture_date][sequence][camera][frame_idx][source_view_index] = {
                'source_view_path': source_view_img_path
            }

        if len(ignored_source_views_indexes) > 0:
            print(f'Asking source view indexes {source_views_indexes} for target view {frame_idx} '
                  f'of sequence {sequence} (length={sequence_length}). \n'
                  f'Ignoring source indexes {ignored_source_views_indexes} as they would be out of bound.')

    return split_data



############################# IMU ###########################
def populate_imu_split_data(absolute_paths, kitti_raw_root_dir, split_data, source_views_indexes):

    oxts_reader = imu_pose_utils.KITTIRawOxts()

    for target_view_path in absolute_paths:
        capture_date, sequence, camera, filename, frame_idx = get_info_from_kitti_raw_path(target_view_path)
        source_views_paths, valid_indexes = get_available_source_views(split_data, target_view_path,
                                                                       source_views_indexes)
        source_views_paths = [kitti_raw_root_dir / p for p in source_views_paths]

        t = oxts_reader.get_transformations_between_pairs(target_view_path, source_views_paths)
        m = oxts_reader.get_magnitude_between_pairs(target_view_path, source_views_paths)

        for i, source_idx in enumerate(valid_indexes):
            data = {
                'imu_pose': t[i],
                'imu_translation_magnitude': m[i]
            }
            split_data[capture_date][sequence][camera][frame_idx][source_idx].update(data)

############################# PnP ###########################
def rgb_read(filename):
    assert Path(filename).exists(), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png

def read_npz_depth(file):
    depth = np.load(file)['velodyne_depth'].astype(np.float32)
    return depth


def populate_pnp_split_data(cam2cam, sparse_depth_root_dir, absolute_paths, kitti_raw_root_dir, split_data,
                            source_views_indexes, threshold_translation=0.15):

    for target_view_path in absolute_paths:

        # paths
        capture_date, sequence, camera, filename, frame_idx = get_info_from_kitti_raw_path(target_view_path)
        source_views_paths, valid_indexes = get_available_source_views(split_data, target_view_path,
                                                                       source_views_indexes)
        source_views_paths = [kitti_raw_root_dir / p for p in source_views_paths]
        depth_path = Path(sparse_depth_root_dir) / \
                     capture_date / sequence / PROJECTED_VELODYNE_DIR / camera / f"{frame_idx:010d}.npz"

        # images
        target_img = Image.open(str(target_view_path))
        source_views_imgs = [Image.open(str(source)) for source in source_views_paths]
        if depth_path.exists():
            sparse_depth = read_npz_depth(str(depth_path))
        else:
            print(f'depth_path {depth_path} not found, setting pose to identity transformation (rot=0°,trans=0)')

        # intrinsics
        K = cam2cam[capture_date][camera.replace('image', 'K')][:3, :3]  # e.g., camera == 'image_02'; replace -> 'K_02'

        for i, source_idx in enumerate(valid_indexes):

            translation_magnitude = 0.0
            vec = np.zeros(6)
            success = False

            if depth_path.exists():
                source_view_img = source_views_imgs[i]
                success, r_vec, t_vec = pnp_pose_utils.get_pose_pnp(np.array(target_img),
                                                                    np.array(source_view_img),
                                                                    sparse_depth,
                                                                    K
                                                                    )

                # success is computation ok and car moved more than threshold
                success = success and np.linalg.norm(t_vec) > threshold_translation

                if success:
                    translation_magnitude = np.linalg.norm(t_vec)
                    vec = np.concatenate([t_vec, r_vec], axis=0).flatten()

            data = {
                'pnp_pose': vec,
                'pnp_translation_magnitude': translation_magnitude,
                'pnp_success': int(success)
            }
            split_data[capture_date][sequence][camera][frame_idx][source_idx].update(data)


@click.command()
@click.argument('kitti_raw_root_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_path', type=click.Path(file_okay=True))
@click.argument('data_split_dir', type=click.Path(file_okay=False))
@click.argument('split_file_names', type=str)
@click.argument('source_views_indexes', type=str)
@click.option('--extend', is_flag=True,
              help="Set this flag to update (extend) an existing split_data.pkl instead of overwriting the whole file")
@click.option('--imu', is_flag=True, help="Set this flag to compute the data from IMU/GPS data")
@click.option('--pnp', type=click.Path(exists=True, file_okay=False),
              help="Provide a path to projected LiDAR images to compute the pose with Perseptive-n-Point method")
@click.option('--pnp_threshold_translation', type=float, default=0.15,
              help="Discard examples where the relative pose magnitude is below the threshold"
                   "(i.e., when the doesn't move more than 'pnp_threshold_translation' meters).")
def main(kitti_raw_root_dir, output_path, data_split_dir, split_file_names, source_views_indexes, extend, imu, pnp,
         pnp_threshold_translation):
    """
    \b
    Create a pickled dictionary at OUTPUT_PATH in which information between taget and source views can be stored:
    - the source views available for each image listed in the split file
    - the relative pose between the source and target views using the IMU or Perspective-n-point w/ LiDAR

    \b
    For example:
    split_data[capture_date][sequence][camera][frame_idx][source_view_index] = {
        'source_view_path': ...,
        'pose_imu': ...,
        'pose_pnp': ...,
    }

    SOURCE_VIEWS_INDEXES is the extent of the temporal context: [-1,1] collects source views at t-1 and t+1.
    """

    assert source_views_indexes[:1] == '[' and source_views_indexes[-1:] == ']', \
        f"Please, give `source_views_indexes` as list format e.g. [-1,1]. What you gave: {source_views_indexes}"
    source_views_indexes = source_views_indexes[1:-1] # remove brackets (first and last char)
    source_views_indexes = source_views_indexes.replace(' ', '').split(',') # remove spaces
    source_views_indexes = list(set([int(str_idx) for str_idx in source_views_indexes])) # remove duplicate

    split_file_names = split_file_names.split(',')

    print(f'Requested source views indexes: {source_views_indexes}')
    print(f'Processing split data for split files: {split_file_names}')
    print(f'Aggregating list of files from split files....')
    relative_paths, capture_dates, seq_dirs, cameras = aggregate_split_files_metadata(data_split_dir, split_file_names)

    kitti_raw_root_dir = Path(kitti_raw_root_dir)

    output_path = Path(output_path)
    if extend:
        if not output_path.exists():
            raise ValueError(f"You asked to extend data but given output_path not found or does not exists: {output_path}")
        print(f'Loading split data from file {output_path} ....')
        with open(output_path, 'rb') as f:
            split_data = pickle.load(f)
    else:
        print(f'Output will be located at: {output_path}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Initialize split data....')
        split_data = get_split_data(kitti_raw_root_dir, relative_paths, capture_dates, seq_dirs, cameras,
                                    source_views_indexes)

    absolute_paths = [Path(kitti_raw_root_dir) / p for p in relative_paths]

    if not (imu or pnp):
        raise ValueError(f"Given output_path not found or doesn't not exists: {output_path}")

    if imu:
        print(f'Append imu pose data to split data dict....')
        populate_imu_split_data(tqdm(absolute_paths), kitti_raw_root_dir, split_data, source_views_indexes)

    if pnp:
        print(f'Append pnp pose data to split data dict....')
        pnp = Path(pnp)
        if not pnp.exists():
            raise ValueError(f"Given path to sparse depth for pnp computation not found or doesn't not exists: {pnp}")
        cam2cam, velo2cam = get_calib_data_for_used_capture_dates(capture_dates, kitti_raw_root_dir)
        populate_pnp_split_data(cam2cam, pnp, tqdm(absolute_paths), kitti_raw_root_dir, split_data,
                                source_views_indexes, pnp_threshold_translation)

    with open(output_path, 'wb') as f:
        pickle.dump(split_data, f)

if __name__ == '__main__':
    main()