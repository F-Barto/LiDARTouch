from logging import getLogger
import numpy as np
from pathlib import Path
import json
import click
import pickle
from tqdm import tqdm

from argoverse.utils.calibration import RING_CAMERA_LIST, get_calibration_config
from argoverse.data_loading.synchronization_database import SynchronizationDB

VEHICLE_CALIBRATION_INFO_FILENAME = 'vehicle_calibration_info.json'
ARGO_SPLIT_NAMES = ['train1', 'train2', 'train3', 'train4', 'val', 'test']

logger = getLogger()


def load_camera_config(calib_filepath, camera_list):
    """
    In the calib file `calib_cam_to_cam.txt` the intrinsics of a camera can be extracted from
    its projective matrix (3x4 matrix)

    Parameters
    ----------
    calib_filepath: str
        The path to the calibration .json file

    camera_list: list
        list of camera for which to load the parameters. Must be in argoverse's CAMERA_LIST

    Returns
    -------
    dict
        configs[camera_name] contains the a argoverse-api CameraConfig with the attributes:
            extrinsic: extrinsic matrix
            intrinsic: intrinsic matrix
            img_width: image width
            img_height: image height
            distortion_coeffs: distortion coefficients

    """
    with open(calib_filepath, "r") as f:
        calib = json.load(f)

    configs = {}
    for camera in camera_list:
        cam_config = get_calibration_config(calib, camera)
        configs[camera] = cam_config
    return configs, calib

def get_split_cam_configs_and_sync_data(argoverse_tracking_root_dir, split_dir, camera_list, source_views_indexes):
    """
    Returns a dictionary giving the intrinsics/extrinsics/resolution of given cameras for each sequence.

    Parameters
    ----------
    split_dir: pathlib Path
        argoverse-tracking split root dir, e.g.: path_to_data/argoverse-tracking/train1

    camera_list: list
        list of camera for which to load the parameters. Must be in argoverse's CAMERA_LIST

    Returns
    -------
    dict
        A dictionary where the key is the log string and the value a dict corresponding to the cameras' parameters
    """

    calibs = {}
    camera_configs = {}
    synchronized_data = []

    db = SynchronizationDB(str(split_dir))
    valid_logs = list(db.get_valid_logs())  # log_ids founds under split_dir

    split = split_dir.stem  # e.g., train1
    for log in tqdm(valid_logs):
        camera_configs[log], calibs[log] = load_camera_config(str(split_dir / log / VEHICLE_CALIBRATION_INFO_FILENAME),
                                                      camera_list)
        synchronized_data += get_synchronized_data(argoverse_tracking_root_dir, db, split, log, camera_list,
                                                   source_views_indexes)

    return calibs, camera_configs, synchronized_data


def get_synchronized_data(argoverse_tracking_root_dir, db, split_name, log, camera_list, source_views_indexes=None):
    """
    returns a list of relative paths for lidar-synched data

    >>> db = SynchronizationDB(...)
    >>> split = 'val'
    >>> log = '5ab2697b-6e3e-3454-a36a-aba2c6f27818'
    >>> camera_list = RING_CAMERA_LIST
    >>> self.get_synchronized_data(db, split, log, camera_list, source_views_indexes=None)[0]
    ['val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/lidar/PC_315972990020197000.ply',
     ('val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_front_center/ring_front_center_315972990006176784.jpg', None),
     ('val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_front_left/ring_front_left_315972990006174448.jpg', None),
     ('val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_front_right/ring_front_right_315972990006174664.jpg', None),
     ('val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_rear_left/ring_rear_left_315972990006176912.jpg', None),
     ('val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_rear_right/ring_rear_right_315972990006174888.jpg', None),
     ('val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_side_left/ring_side_left_315972990006174840.jpg', None),
     ('val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_side_right/ring_side_right_315972990006175456.jpg', None)]
    """

    logger.info(f'Collecting the files in the split {split_name}'
                         f' that are synchronized with LiDAR and have requested source views...')

    synchronized_data = []

    for lidar_timestamp in db.per_log_lidartimestamps_index[log]:

        lidar_prefix = f"{split_name}/{log}/lidar/"
        lidar_point_cloud_path = lidar_prefix + f"PC_{lidar_timestamp}.ply"

        camera_images_filepaths = []
        for camera_name in camera_list:
            camera_prefix = f"{split_name}/{log}/{camera_name}/"
            closest_cam_timestamp = db.get_closest_cam_channel_timestamp(lidar_timestamp, camera_name, log)
            image_filepath = f"{camera_name}_{closest_cam_timestamp}.jpg"
            camera_images_filepaths.append(camera_prefix + image_filepath)

        if len(camera_images_filepaths) == len(camera_list):
            # we found a synched image for each camera
            target_and_source_views_paths = get_valid_samples_and_source_views_paths(
                argoverse_tracking_root_dir, db, log, camera_images_filepaths, source_views_indexes
            ) # return None for source views path if no source views requested

            if len(target_and_source_views_paths) == len(camera_images_filepaths):
                # we found the requested source views for all target view of each camera
                synchronized_paths = [lidar_point_cloud_path] + target_and_source_views_paths
                synchronized_data.append(synchronized_paths)

        else:
            logger.warning(f"Split {split_name} | log: {log} |"
                                    f"Not all cameras are synchronized with LiDAR at {lidar_timestamp} ")

    logger.info('Files collected and requested source-views registered.')

    return synchronized_data

def get_valid_samples_and_source_views_paths(argoverse_tracking_root_dir, db, log, img_paths_by_cam, source_views_indexes):
    """
    Discard the samples that don't have avaible source views, where the source views are defined by source_indexes.

    Parameters
    ----------
    img_paths: list of str
        A list of absolute paths to the data samples.
        Each element of img_paths is of the form:
        path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png

    Returns
    -------
    list of tuple
        A curated list of absolute paths to the data samples ensured to have source views.
        each element of the list is a tuple (target_img_path, [source_view1_img_path, ..., source_viewN_img_path])
    """

    if source_views_indexes is None or len(source_views_indexes) < 1:
        return [(img_path, None) for img_path in img_paths_by_cam]

    img_paths_and_source_views = []

    for img_path in img_paths_by_cam:

        # img_path == path_to_data/train1/6f153f9c-edc5-389f-ac6f-40705c30d97e/ring_front_center/ring_front_center_315966434380895928.jpg
        img_path = Path(img_path)
        assert (argoverse_tracking_root_dir / img_path).exists(), img_path

        camera_name = img_path.parent.stem  # ring_front_center
        camera_timestamps = db.per_log_camtimestamps_index[log][camera_name]

        img_timestamp = img_path.stem[len(camera_name) + 1:]  # 315972990006176784
        timestamp_idx = np.searchsorted(camera_timestamps, img_timestamp)

        # checks if any source source view index falls out of the sequence's bounds
        log_length = len(camera_timestamps)
        if timestamp_idx + source_views_indexes[0] < 0 or timestamp_idx + source_views_indexes[-1] >= log_length:
            # remember that source_views_indexes is in ascending order
            logger.warning('One or more source views from the ones requested is out of bound.\n'
                           f'\tIndexes requested: {source_views_indexes} -> '
                           f'Ignoring frame {timestamp_idx} of timestamp {img_timestamp} from log {log}.\n'
                           f'\tThe log {log} is of length {log_length}')
            continue

        # checks if any source view is missing
        missing_source_view = False
        source_view_paths = get_source_view_paths_from_img_path_and_indexes(db, log, img_path, source_views_indexes)
        for source_view_index, source_view_img_path in zip(source_views_indexes, source_view_paths):
            if not (argoverse_tracking_root_dir / source_view_img_path).exists():
                logger.warning(f'Frame of timestamp {img_timestamp} from log {log} have the '
                               f'source view of idx {source_view_index} missing from the ones requested.\n'
                               f'\tIndexes requested: {source_views_indexes} -> '
                               f'Ignoring frame idx {timestamp_idx}, ts {img_timestamp} from log {log}.\n')
                missing_source_view = True
                # not breaking the loop so that we know all the source views that are unavailable

        if not missing_source_view:
            img_paths_and_source_views.append((str(img_path), source_view_paths))

    return img_paths_and_source_views

def get_source_view_paths_from_img_path_and_indexes(db, log, img_path, source_views_indexes):
    """
    Parameters
    ----------
    img_path: str
        Absolute or relative path to the data sample in argoverse-tracking format.
        e.g.: path_to_dataset_root_dir/train1/6f153f9c-edc5-389f-ac6f-40705c30d97e/ring_front_center/ring_front_center_315966434380895928.jpg
    source_views_indexes : list of int
        The relative indexes to sample from the neighbouring views of the target view.
        It is expected that the list is in ascending order and does not contains 0 (corresponding to the target).
        For example, source_indexes=[-1,1] will load the views at time t-1, t, t+1

    Returns
    -------
    int
        Absolute or relative path of the requested source view w.r.t to img_path

    Example
    -------
    img_path = "val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_front_center/ring_front_center_315972990006176784.jpg"
    source_views_indexes = [-1,1]
    get_source_view_paths_from_indexes_and_img_path(img_path,source_views_indexes)
    >> ["val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_front_center/ring_front_center_315966434347596072.jpg",
    >> "val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_front_center/ring_front_center_315966434414195816.jpg"]
    """

    img_path = Path(img_path)

    camera_name = img_path.parent.stem # ring_front_center
    camera_timestamps = db.per_log_camtimestamps_index[log][camera_name]

    img_timestamp = img_path.stem[len(camera_name) + 1 :] # 315972990006176784
    timestamp_idx = np.searchsorted(camera_timestamps, img_timestamp)

    source_views_img_paths = []
    for source_view_index in source_views_indexes:
        source_view_frame_timestamp = camera_timestamps[timestamp_idx + source_view_index]
        source_view_img_path = img_path.parent / f'{camera_name}_{source_view_frame_timestamp}.jpg'
        source_views_img_paths.append(str(source_view_img_path))

    return source_views_img_paths

def collect_cam_configs_and_sync_data(argoverse_tracking_root_dir, camera_list, split_name, source_views_indexes,
                                      return_calib=False):

    """
    :param argoverse_tracking_root_dir:
    :param camera_list:
    :param split_name:
    :param source_views_indexes:
    :return:

    a camera_configs dict indexed by log name
    a list of relative paths for lidar-synched data with source views as in get_synchronized_data
    """

    argoverse_tracking_root_dir = Path(argoverse_tracking_root_dir).expanduser()
    assert argoverse_tracking_root_dir.exists(), argoverse_tracking_root_dir
    argoverse_tracking_root_dir = argoverse_tracking_root_dir

    assert split_name in ARGO_SPLIT_NAMES

    split_dir = argoverse_tracking_root_dir / split_name
    assert split_dir.exists(), split_dir
    calibs, cc, paths = get_split_cam_configs_and_sync_data(argoverse_tracking_root_dir, split_dir, camera_list,
                                                    source_views_indexes)
    camera_configs, samples_and_source_views_paths = cc, paths

    logger.info(f'Dataset for split {split_name} ready.\n\n' + '-' * 90 + '\n\n')

    if return_calib:
        return  calibs, camera_configs, samples_and_source_views_paths
    else:
        return camera_configs, samples_and_source_views_paths

@click.command()
@click.argument('argo_tracking_root_dir', type=click.Path(exists=True, file_okay=False)) # .../argoverse-tracking/ under which you can find "train1", "train2", ...
@click.argument('output_base_dir', type=click.Path(exists=True, file_okay=False))
def main(argo_tracking_root_dir, output_base_dir):
    print('Synchronizing data....')
    print("INPUT DIR: ", argo_tracking_root_dir)

    output_base_dir = Path(output_base_dir).expanduser().resolve()
    output_file_path = str(output_base_dir / f"argoverse_ring_synchronized_data.pkl")

    print("OUTPUT File: ", output_file_path)

    output = {}

    for split_name in ARGO_SPLIT_NAMES:

        print(f"processing {split_name} split...")

        split_data = collect_cam_configs_and_sync_data(argo_tracking_root_dir, RING_CAMERA_LIST,
                                                       split_name, source_views_indexes=None, return_calib=True)

        output[split_name] = {
            'calibs': split_data[0],
            'camera_configs': split_data[1],
            'camera_list': RING_CAMERA_LIST,
            'source_views_indexes': None,
            'samples_paths': split_data[2],
        }

    with open(output_file_path, 'wb') as f:
        pickle.dump(output, f)

    print('argoverse data Lidar-synchronized for ring cameras.')



if __name__ == '__main__':
    main()