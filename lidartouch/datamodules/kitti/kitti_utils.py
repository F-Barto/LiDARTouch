import numpy as np
from pathlib import Path

from typing import List, Tuple, Union


# Constants
CAM2CAM_CALIB_FILENAME = 'calib_cam_to_cam.txt'
VELO2CAM_CALIB_FILENAME = 'calib_velo_to_cam.txt'


def get_info_from_kitti_raw_path(img_path: Union[Path, str]) -> Tuple[str, str, str, str, int]:
    """
    Extracts the information from a given KITTI raw path and return a tuple with
    the information contained in the path:
        (capture_date, sequence, camera, filename, frame_idx)

    Args:
        img_path : Absolute or relative path to the data sample in KITTI Raw format.

    Example:
        img_path = '2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png'
        get_info_from_relative_path(img_path)
        >>>     ('2011_09_26', '2011_09_26_drive_0001_sync', 'image_02', '0000000000', 0)
    """
    img_path = Path(img_path)

    capture_date = img_path.parents[3].name
    sequence = img_path.parents[2].name
    camera = img_path.parents[1].name
    filename = img_path.stem
    frame_idx = int(filename)

    return capture_date, sequence, camera, filename, frame_idx

def get_sequences_lengths(kitti_raw_root_dir: Union[Path, str], sequences: List[str], cameras: List[str]) -> dict:
    """
    Returns a dictionary giving the number of frames for each sequences in split_file

    Args:
        kitti_raw_root_dir : absolute path to the kitti raw directory
        sequences : list of sequence dir you want to obtain the length
        cameras : list of cameras (e.g., ['image_02', 'image_03'])

    Returns:
        A dictionary where the key is the directory name of a sequence and the value
        the number of frames it contains, i.e., the lenth of the sequence
    """
    kitti_raw_root_dir = Path(kitti_raw_root_dir)

    sequence_lengths = {}
    for seq in sequences:
        # e.g., seq_dir = 2011_09_26_drive_0001_sync
        capture_date = seq[:10]
        sequence_lengths[seq] = {}
        for camera in cameras:
            data_dir = kitti_raw_root_dir / capture_date / seq / camera / 'data'
            assert data_dir.exists(), data_dir
            sequence_lengths[seq][camera] = len([f for f in data_dir.iterdir()])

    return sequence_lengths


def aggregate_split_files_metadata(data_split_dir: Union[Path, str], split_file_names: List[str]) -> Tuple[List[str], ...]:
    """
    Take all given split files and return a list of all relative path apperaing
    in the given split files without duplicates.

    Also return the list of all capture dates, sequences and cameras present in the split files

    A split file is a .txt file where each line is a relative path to a KITTI Raw image.
    Example of KITTI raw relative path:
        2011_09_26/2011_09_26_drive_0022_sync/image_02/data/0000000532.png

    Args:
        data_split_dir: Path to directory in which the split files are located
        split_file_names: list of string containing the name of the split files

    Return:
        relative_paths: list of all relative path appearing in the given split files without duplicates.
        capture_dates: list of all capture dates
        sequences: list of all sequences
        cameras: list of all capture
    """

    data_split_dir = Path(data_split_dir)

    arr = []
    for split_file_name in split_file_names:
        split_file_path = data_split_dir / split_file_name

        # each row of arr is of the form
        # ['2011_09_26', '2011_09_26_drive_0048_sync', 'image_02', 'data', '0000000085.png']
        tmp_arr = np.genfromtxt(split_file_path, delimiter='/', dtype=str)
        arr.append(tmp_arr)

    arr = np.concatenate(arr)

    # discard duplicates and sort to avoid random ordering of paths
    relative_paths = np.unique(np.apply_along_axis(lambda d: '/'.join(d), 1, arr))
    # discard duplicates and sort to avoid random ordering of paths
    relative_paths = sorted(list(relative_paths))

    capture_dates = list(np.unique(arr[:, 0]))
    sequences = list(np.unique(arr[:, 1]))
    cameras = list(np.unique(arr[:, 2]))

    return relative_paths, capture_dates, sequences, cameras

def get_split_file_metadata(split_file: Union[Path, str]):
    split_file = Path(split_file)
    return aggregate_split_files_metadata(split_file.parent, [split_file.name])


def get_available_source_views(split_data, target_view_path, source_views_indexes):
    source_views_paths = []
    valid_indexes = []
    capture_date, sequence, camera, filename, frame_idx = get_info_from_kitti_raw_path(target_view_path)
    for source_view_index in source_views_indexes:
        source_view_elems = split_data[capture_date][sequence][camera][frame_idx].get(source_view_index, None)
        if source_view_elems:
            source_views_paths.append(source_view_elems['source_view_path'])
            valid_indexes.append(source_view_index)
    return source_views_paths, valid_indexes


############################################################################
####################   PATHS AND CALIB DATA  ###############################
############################################################################


def read_calib_file(calib_file_path: Union[Path, str]) -> dict:
    """
    In calibration files of kitti raw, all lines are in the format:
    <matrix or data name>: <data>

    example (first three lines of calib_cam_to_cam.txt):
        calib_time: 09-Jan-2012 13:57:47
        corner_dist: 9.950000e-02
        S_00: 1.392000e+03 5.120000e+02

    We create a dict by splitting the lines at the first colon character
    while handeling the `calib_time` case where multiple colon characters can be
    found in the same line.

    Returns:
        calibs: the calibration data in a dict (e.g., calibs['S_00'] == '1.392000e+03 5.120000e+02')
            note that the values are stored as string.
    """

    with Path(calib_file_path).open() as calib_file:
        lines = calib_file.readlines()
        calibs = {l.split(":")[0]: ':'.join(l.split(":")[1:]) for l in lines}

    return calibs


def process_cam2cam_data(cam_to_cam_calib_file_path: Union[Path, str], homogeneous: bool = False) -> dict:
    """
    Process the cam to cam calibration data from calib file.

    Note that, in the calib file `calib_cam_to_cam.txt` the first two lines are:
        1. the data and time at which the calibration has been made
        2. corner distance
    After that, the lines are intrinsics/extrinsics/projection matrixes

    from KITTI Raw devkit:
        - S_xx: 1x2 size of image xx before rectification
        - K_xx: 3x3 calibration matrix of camera xx before rectification
        - D_xx: 1x5 distortion vector of camera xx before rectification
        - R_xx: 3x3 rotation matrix of camera xx (extrinsic)
        - T_xx: 3x1 translation vector of camera xx (extrinsic)
        - S_rect_xx: 1x2 size of image xx after rectification
        - R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
        - P_rect_xx: 3x4 projection matrix after rectification

    Returns:
        calibs: the calibration data in a dict where values are numpy array in correct shape
    """

    calibs = read_calib_file(cam_to_cam_calib_file_path)
    calibs.pop('calib_time')
    calibs.pop('corner_dist')

    KEYPREFIX_TO_SHAPE = {
        'S': (1, 2),
        'K': (3, 3),
        'D': (1, 5),
        'R': (3, 3),
        'T': (3, 1),
        'P': (3, 4),
    }

    HOMOGENEOUS_KEYS_PREFIXES = ['R', 'K', 'T']

    for key, string_value in calibs.items():
        key_prefix = key[0]
        arr_value = np.array(string_value.split()).astype(float)
        arr_value = arr_value.reshape(KEYPREFIX_TO_SHAPE[key_prefix])

        if homogeneous and key_prefix in HOMOGENEOUS_KEYS_PREFIXES:
            h_arr_value = np.eye(4)
            if arr_value.shape == (3, 1):
                # T_xx 3x1 -> 4x4
                h_arr_value[0:3, 3:4] = arr_value
                arr_value = h_arr_value
            else:
                # 3x3 -> 4x4
                h_arr_value[0:3, 0:3] = arr_value
                arr_value = h_arr_value

        calibs[key] = arr_value

    return calibs


def process_velo2cam_data(velo2cam_calib_file_path: Union[Path, str], homogeneous: bool = False):
    """
    In the calib file `calib_velo_to_cam.txt` the extrinsics of a camera can be extracted from
    the rotation matrix R: 3x3 and the  translation vector T: 3x1

    R|T takes a point in Velodyne coordinates and transforms it into the
    coordinate system of the left video camera. Likewise it serves as a
    representation of the Velodyne coordinate frame in camera coordinates.

    Args:
        velo2cam_calib_file_path : path to the relevant calibration file
        homogeneous : set to True if you want extrinsics to be returned as homogeneous (4x4 in place of 3x4)

    Returns:
        RT : numpy array, 3x4 (4x4 if homogeneous is True)

    """
    calibs = read_calib_file(velo2cam_calib_file_path)

    transformation_matrix = np.eye(4)

    R = np.array(calibs['R'].split()).astype(float).reshape((3, 3))
    T = np.array(calibs['T'].split()).astype(float).reshape((3, 1))

    transformation_matrix[0:3, 0:3] = R
    transformation_matrix[0:3, 3:4] = T

    if not homogeneous:
        return transformation_matrix[0:3, :]  # discard last line

    return transformation_matrix


def get_calib_data_for_used_capture_dates(capture_dates: List[str], kitti_raw_root_dir: Union[Path, str]) -> Tuple[dict, ...]:
    """
    Return the calibration data for all the given captures dates

    Args:
        capture_dates : list of capture dates for which to load calibration data
        kitti_raw_root_dir : absolute path to the kitti raw directory

    Returns:
        cam2cam : camera to camera calibration data in a dict
        velo2cam: velodyne LIDAR to camera calibration data in a dict
    """

    cam2cam = {}
    velo2cam = {}

    kitti_raw_root_dir = Path(kitti_raw_root_dir)

    for capture_date in capture_dates:
        cam2cam_calib_path = kitti_raw_root_dir / capture_date / CAM2CAM_CALIB_FILENAME
        velo2cam_calib_path = kitti_raw_root_dir / capture_date / VELO2CAM_CALIB_FILENAME

        assert cam2cam_calib_path.exists(), f"Can't find expected calib file {cam2cam}"
        assert velo2cam_calib_path.exists(), f"Can't find expected calib file {velo2cam}"

        cam2cam_data = process_cam2cam_data(cam2cam_calib_path, homogeneous=True)
        cam2cam[capture_date] = cam2cam_data

        velo2cam_data = process_velo2cam_data(velo2cam_calib_path, homogeneous=True)
        velo2cam[capture_date] = velo2cam_data

    return cam2cam, velo2cam