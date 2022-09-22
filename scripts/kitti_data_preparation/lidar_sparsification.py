import click
from tqdm import  tqdm
from joblib import Parallel, delayed
from functools import partial

from pathlib import Path
import numpy as np

from pykitti.utils import load_oxts_packets_and_poses, load_velo_scan

'''
python ./lidar_sparsification.py \
kitti_raw_root_dir \
output_dir \
data_split_dir \
'split_file1,split_file2' \
--downsample_indexes='5,7,9,11'

python ./lidar_sparsification.py \
/home/clear/fbartocc/data/KITTI_raw/  \
/home/clear/fbartocc/working_data/KITTI/MC_sparse_lidar/ \
/home/clear/fbartocc/depth_project/LiDARTouch/data_splits \
'INFER_sequences_image_02.txt,eigen_train_files.txt,filtered_eigen_val_files.txt,filtered_eigen_test_files.txt' \
--downsample_factor=16
'''

###########################################################################
####################   motion compensation  ###############################
###########################################################################

def get_correct_d_oxts(velo_path):
    """
    velodyne file path in format:
    kitti_raw_root_dir/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000105.bin
    """

    velo_path = Path(velo_path)

    assert velo_path.exists(), velo_path
    real_id = int(velo_path.stem)

    oxts0_filename = velo_path.parents[2] / 'oxts/data' / '0000000000.txt'
    oxts1_filename = velo_path.parents[2] / 'oxts/data' / f'{real_id:010d}.txt'
    oxts2_filename = velo_path.parents[2] / 'oxts/data' / f'{(real_id + 1):010d}.txt'

    assert oxts0_filename.exists(), f"can't find oxts0 (corrupted data ?): {oxts0_filename}"
    assert oxts1_filename.exists(), f"can't find oxts1 (corrupted data ?): {oxts1_filename}"
    assert oxts2_filename.exists(), f"can't find oxts2 (certainly reached end of sequence): {oxts2_filename}"

    # Poses are given in an East-North-Up coordinate system
    # whose origin is the first GPS position (oxts0).
    oxts1 = load_oxts_packets_and_poses([oxts0_filename, oxts1_filename])[1]
    oxts2 = load_oxts_packets_and_poses([oxts0_filename, oxts2_filename])[1]

    return oxts1, oxts2


def get_relative_motion(oxts1, oxts2):
    """
    Gets the relative dx, dy, dz, dyaw of b2 relative to b1
    dyaw is represented in radians, not degrees
    """
    T_b1_w = inverse_rigid_transformation(oxts1.T_w_imu)
    T_b1_b2 = T_b1_w.dot(oxts2.T_w_imu)
    tx, ty, tz = T_b1_b2[0, 3], T_b1_b2[1, 3], T_b1_b2[2, 3]
    roll = oxts2.packet.roll - oxts1.packet.roll
    pitch = oxts2.packet.pitch - oxts1.packet.pitch
    yaw = oxts2.packet.yaw - oxts1.packet.yaw
    return tx, ty, tz, roll, pitch, yaw


def inverse_rigid_transformation(T_a_b):
    """
    Computes the inverse transformation T_b_a from T_a_b
    """
    R_a_b = T_a_b[0:3, 0:3]
    t_a_b = T_a_b[0:3, 3]
    R_b_a = np.transpose(R_a_b)
    t_b_a = - R_b_a.dot(t_a_b).reshape(3, 1)
    T_b_a = np.vstack((np.hstack([R_b_a, t_b_a]), [0, 0, 0, 1]))
    return T_b_a


def transform_from_xyz_euler(tx, ty, tz, roll, pitch, yaw):
    s = np.sin(yaw)
    c = np.cos(yaw)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]]
                 )
    t = np.array([tx, ty, tz])
    return R, t


def compensate_motion(scanline, scan_time, tx, ty, tz, roll, pitch, yaw):
    # x: positive forward
    # y: positive to the left
    # rotation angle: the lidar is spinning counter-clockwise.
    # need to compensate for both positional change and angular change of the vehicle over time
    for j in range(len(scanline)):
        ratio = 0.5 - scan_time[j]
        R, t = transform_from_xyz_euler(tx * ratio, ty * ratio, tz * ratio, roll * ratio, pitch * ratio, yaw * ratio)
        raw_coordinate = scanline[j, :3].reshape(3, 1)
        scanline[j, :3] = R.dot(raw_coordinate).reshape(3) + t
    return scanline

############################################################################
####################   PATHS AND CALIB DATA  ###############################
############################################################################

def get_meta(line):
    line = line.split('/')
    filename = line[-1]
    capture_date = line[0]
    data_dir = line[1]
    camera = line[2]

    return capture_date, data_dir, camera, filename

def get_captures_date_from_split_file(split_file_path):
    """
    Extract all the capture dates used in given split file

    Under the root dir, KITTI raw is organized as follow
    {capture_date}/{capture_date}_drive_{sequence_idx:04d}_sync/image_02/data/{frame_idx:010d}.png
    e.g., 2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png
    thus, each line of the split file is assumed to have the same formatting

    split_file_path : str
        absolute path to the split file to process

    Returns:
        list of all the capture dates used in given split file
    """

    arr = np.genfromtxt(split_file_path, delimiter='/', dtype=str)
    # each row of arr is of the form
    # ['2011_09_26', '2011_09_26_drive_0048_sync', 'image_02', 'data', '0000000085.png']
    used_capture_dates = np.unique(arr[:, 0])

    return sorted(list(used_capture_dates))


def read_calib_file(calib_file_path):
    """
    In the calib file `calib_cam_to_cam.txt` the first two lines are:
        1. the data and time at which the calibration has been made
        2. corner distance
    After that, the lines are intrinsics/extrinsics/projection matrixes

    All lines are in the format:
    <matrix or data name>: <data>

    example (first three lines):
        calib_time: 09-Jan-2012 13:57:47
        corner_dist: 9.950000e-02
        S_00: 1.392000e+03 5.120000e+02

    We create a dict by splitting the lines at the first colon character
    while handeling the `calib_time` case where multiple colon characters can be
    found in the same line

    Returns
    -------
    dict
        the calibration data in a dict (e.g., calibs['S_00'] == '1.392000e+03 5.120000e+02')
    """

    with Path(calib_file_path).open() as calib_file:
        lines = calib_file.readlines()
        calibs = {l.split(":")[0]: ':'.join(l.split(":")[1:]) for l in lines}

    return calibs


def get_rect_img_size(calibs, camera_id):
    """
    Extract from the calib file `calib_cam_to_cam.txt` the size of image <camera_id> after rectification

    Parameters
    ----------
    calibs : dict
        data from calib_cam_to_cam.txt extracted with read_calib_file()
    camera_id : int
        the camera to get the resolution for (expected value in 0,1,2,3)

    Returns
    -------
    tuple
        height and width of image

    """

    # first convert to float THEN to int otherwise error: invalid literal for int() with base 10: '1.242000e+03'
    w, h = np.array(calibs[f"S_rect_0{camera_id}"].split()).astype(float).astype(int)

    return h, w


def get_projective_matrix(calibs, projective_matrix):
    """
    In the calib file `calib_cam_to_cam.txt` the intrinsics of a camera can be extracted from
    its projective matrix P_rect_xx: 3x4 projection matrix after rectification


    Parameters
    ----------
    calibs : dict
        data from calib_cam_to_cam.txt extracted with read_calib_file()

    projective_matrix : str
        the name of the projective matrix in format P_rect_xx (e.g., P_rect_02)

    Returns
    -------
    np array
        3x4 projection matrix after rectification

    """
    P = np.array(calibs[projective_matrix].split()).astype(float).reshape((3, 4))

    return P


def get_R_rect_00(calibs, homogeneous=False):
    """

    Parameters
    ----------
    calibs : dict
        data from calib_cam_to_cam.txt extracted with read_calib_file()
    homogeneous : bool
        set to True if you want the matrix to be returned as homogeneous (4x4 in place of 3x3)

    Returns
    -------
    np array
        R_rect_00:  3x3 rectifying rotation to make image planes co-planar
                    cam 0 coordinates -> rectified cam 0 coord.

    """
    R_rect = np.array(calibs['R_rect_00'].split()).astype(float).reshape((3, 3))

    if homogeneous:
        h_R_rect = np.pad(R_rect, ((0, 1), (0, 1)))
        h_R_rect[-1, -1] = 1
        return h_R_rect

    return R_rect


def get_extrinsics(calib_file_path, homogeneous=False):
    """
    In the calib file `calib_velo_to_cam.txt` the extrinsics of a camera can be extracted from
    the rotation matrix R: 3x3 and the  translation vector T: 3x1

    Parameters
    ----------
    calib_file_path : str
        path to the relevant calibration file
    homogeneous: bool
        set to True if you want extrinsics to be returned as homogeneous (4x4 in place of 3x4)

    Returns
    -------
    np array
        R | T: 3x4 (4x4 if homogeneous is True)

    """
    calibs = read_calib_file(calib_file_path)

    R = np.array(calibs['R'].split()).astype(float).reshape((3, 3))
    T = np.array(calibs['T'].split()).astype(float).reshape((3, 1))

    RT = np.concatenate([R, T], axis=1)

    if homogeneous:
        return np.pad(RT, ((0, 1), (0, 0)))

    return RT


def get_cam_projection_data(capture_dates, kitti_root_dir: Path, camera_ids=[2, 3]):
    """
    Return all the calibration data for all captures dates as multiple dicts

    Parameters
    ----------
    capture_dates : list
        captures dates for which to load calibration data

    kitti_root_dir : str
        absolute path to the kitti raw directory

    camera_ids : list of int or int
        the ids of the cameras for which to load calib datas

    Returns
    -------
    img_sizes : dict
        height and width for each capture date

    projective_matrixes : dict
        projective matrix for each capture date

    R_rect_00s : dict
        R_rect_00 for each capture date
    """
    if isinstance(camera_ids, int): camera_ids = [camera_ids]

    projective_matrixes = {}
    R_rect_00s = {}
    img_sizes = {}

    for capture_date in capture_dates:
        projective_matrixes[capture_date] = {}
        for camera_id in camera_ids:
            assert camera_id in range(4), f"Given camera_id is not between 0 and 4: {camera_id}"
            calib_cam_to_cam_path = kitti_root_dir / capture_date / 'calib_cam_to_cam.txt'
            assert calib_cam_to_cam_path.exists(), f"Can't find expected calib file: {calib_cam_to_cam_path}"
            calib_cam_to_cam = read_calib_file(calib_cam_to_cam_path)
            height, width = get_rect_img_size(calib_cam_to_cam, camera_id=camera_id)
            P_rect = get_projective_matrix(calib_cam_to_cam, f'P_rect_0{camera_id}')
            R_rect_00 = get_R_rect_00(calib_cam_to_cam, homogeneous=True)

            img_sizes[capture_date] = (height, width)
            projective_matrixes[capture_date][f'image_0{camera_id}'] = P_rect
            R_rect_00s[capture_date] = R_rect_00

    return img_sizes, projective_matrixes, R_rect_00s


def get_extrinsics_for_used_capture_dates(capture_dates, kitti_root_dir: Path):
    """
    Return the extrinsics data for all the given captures dates

    Parameters
    ----------
    capture_dates : list
        captures dates for which to load calibration data

    kitti_root_dir : str
        absolute path to the kitti raw directory

    Returns
    -------
    extrinsics : dict
        camera extrinsics for each capture date
    """

    extrinsics = {}

    for capture_date in capture_dates:
        calib_velo_to_cam_path = kitti_root_dir / capture_date / 'calib_velo_to_cam.txt'
        assert calib_velo_to_cam_path.exists(), f"Can't find expected calib file {calib_velo_to_cam_path}"
        extrinsic_matrix = get_extrinsics(calib_velo_to_cam_path, homogeneous=True)

        extrinsics[capture_date] = extrinsic_matrix

    return extrinsics


########################################################################
####################   PROJECTION UTILS  ###############################
########################################################################


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


def wrap_to_0_360(deg):
    while True:
        indices = np.nonzero(deg < 0)[0]
        if len(indices) > 0:
            deg[indices] = deg[indices] + 360
        else:
            break

    deg = ((100 * deg).astype(int) % 36000) / 100.0
    return deg


def downsample_scan(velo, tx, ty, tz, roll, pitch, yaw, downsample_factor=None, downsample_indexes=None):
    """
    Downsample HDL-64 scans to the target_scan
    """

    assert (downsample_factor is not None) != (downsample_indexes is not None)
    'Choose either downsample_factor or downsample_indexes'

    x, y, z, r = velo[:, 0], velo[:, 1], velo[:, 2], velo[:, 3]
    # angles between the start of the scan (towards the rear)
    horizontal_degree = np.rad2deg(np.arctan2(y, x))
    horizontal_degree = wrap_to_0_360(horizontal_degree)

    scan_breakpoints = np.nonzero(np.diff(horizontal_degree) < -180)[0] + 1
    scan_breakpoints = np.insert(scan_breakpoints, 0, 0)
    scan_breakpoints = np.append(scan_breakpoints, len(horizontal_degree) - 1)
    num_scans = len(scan_breakpoints) - 1

    # note that sometimes not all 64 scans show up in the image space
    if downsample_indexes is not None:
        assert len(downsample_indexes) > 0
        indexes = downsample_indexes
    elif downsample_factor is not None and downsample_factor > 1:
        indexes = range(num_scans - downsample_factor // 2, -1, -downsample_factor)
    else:
        indexes = range(num_scans - 1, -1, -1)

    assert num_scans <= 65, f"invalid number of scanlines (should be <=64): {num_scans}"

    downsampled_velo = np.zeros(shape=[0, 4])
    for i in indexes:
        start_index = scan_breakpoints[i]
        end_index = scan_breakpoints[i + 1]
        scanline = velo[start_index:end_index, :]
        # the start of a scan is triggered at 180 degree
        scan_time = wrap_to_0_360(horizontal_degree[start_index:end_index] + 180) / 360
        scanline = compensate_motion(scanline, scan_time, tx, ty, tz, roll, pitch, yaw)
        downsampled_velo = np.vstack((downsampled_velo, scanline))
    assert downsampled_velo.shape[0] > 0, "downsampled velodyne has 0 measurements"

    return downsampled_velo


def downsample_motion_comp_lidar_pc(velodyne_path, downsample_factor=None, downsample_indexes=None):
    assert velodyne_path.exists(), f"Veoldyne file does not exists: {velodyne_path}"
    pc_data = np.fromfile(velodyne_path, dtype=np.float32).reshape((-1, 4))
    oxts1, oxts2 = get_correct_d_oxts(velodyne_path)
    tx, ty, tz, roll, pitch, yaw = get_relative_motion(oxts1, oxts2)
    downsampled_scan = downsample_scan(pc_data, tx, ty, tz, roll, pitch, yaw,
                                       downsample_factor=downsample_factor,
                                       downsample_indexes=downsample_indexes)

    return downsampled_scan




def project_and_save(img_sizes, projective_matrixes, R_rect_00s, extrinsics, kitti_raw_root_dir,
                     output_sparse_depth_dir, downsample_factor, downsample_indexes, line):

    capture_date, data_dir, camera, filename = get_meta(line)

    height, width = img_sizes[capture_date]
    P_rect = projective_matrixes[capture_date][camera]
    R_rect_00 = R_rect_00s[capture_date]

    extrinsic_matrix = extrinsics[capture_date]

    velodyne_dir = kitti_raw_root_dir / capture_date / data_dir / 'velodyne_points/data'
    pc_file_path = velodyne_dir / (filename.split('.')[0] + '.bin')

    try:
        sparsified_data = downsample_motion_comp_lidar_pc(pc_file_path,
                                                          downsample_factor=downsample_factor,
                                                          downsample_indexes=downsample_indexes)
        sparsified_pts_3d, sparsified_pts_reflectances = sparsified_data[:, :3], sparsified_data[:, -1]
    except AssertionError as e:
        print(f"Error at file: {str(pc_file_path)} \nAdditional info: {e}")
        lidar_projected = np.zeros((height, width))
        # saving an array full of zeros
        output_base_dir = output_sparse_depth_dir / capture_date / data_dir / 'proj_depth/velodyne' / camera
        output_base_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_base_dir / (filename.split('.')[0] + '.npz')
        np.savez(output_path, velodyne_depth=lidar_projected, pc_data=np.array([]))

        return pc_file_path

    h_pts_3d = np.pad(sparsified_pts_3d, ((0, 0), (0, 1)), constant_values=1)

    uv_cam = R_rect_00 @ extrinsic_matrix @ h_pts_3d.T
    uv_cam = uv_cam

    uv = P_rect @ uv_cam

    uv[0:2, :] /= uv[2, :]
    uv = uv.T
    uv = uv[:, :2]

    valid_pts_bool = determine_valid_cam_coords(uv, uv_cam, width, height)

    uv = uv[valid_pts_bool].astype(int)
    uv_cam = uv_cam.T[valid_pts_bool] # sparsified pc in camera coordinates

    lidar_projected = np.zeros((height, width))
    lidar_projected[uv[:, 1], uv[:, 0]] = uv_cam.T[2]

    sparsified_pts_reflectances = sparsified_pts_reflectances[valid_pts_bool]
    uv_cam[:, -1] = sparsified_pts_reflectances  # replace homogeneous dim w/ reflectance data

    # saving
    output_base_dir = output_sparse_depth_dir / capture_date / data_dir / 'proj_depth/velodyne' / camera
    output_base_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_base_dir / (filename.split('.')[0] + '.npz')

    np.savez(output_path, velodyne_depth=lidar_projected, pc_data=uv_cam)

@click.command()
@click.argument('kitti_raw_root_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.argument('data_split_dir', type=click.Path(file_okay=False))
@click.argument('split_file_names', type=str)
@click.option('--downsample_indexes', type=str)
@click.option('--downsample_factor', type=int)
def main(kitti_raw_root_dir, output_dir, data_split_dir, split_file_names, downsample_indexes=None, downsample_factor=None):

    print('Preprocessing data....')
    print("KITTI RAW DIR: ", kitti_raw_root_dir)
    print('arg split_file_names ', split_file_names)
    split_file_names = split_file_names.replace(' ','').split(',')
    print('processed split_file_names ', split_file_names)

    assert (downsample_factor is not None) != (downsample_indexes is not None), \
        'Choose either downsample_factor or downsample_indexes'

    kitti_raw_root_dir = Path(kitti_raw_root_dir)
    output_dir = Path(output_dir)
    data_split_dir = Path(data_split_dir)
    if downsample_indexes is not None:
        print("Selected indexes: ", downsample_indexes)
        str_downsample_indexes = downsample_indexes.replace(',','_')
        output_sparse_depth_dir = output_dir / f'beams_{str_downsample_indexes}'
        downsample_indexes = np.array(downsample_indexes.split(',')).astype(int)
    if downsample_factor is not None:
        print("Downsampling factor: ", downsample_factor)
        output_sparse_depth_dir = output_dir / f'factor_{downsample_factor}'

    print("OUTPUT DIR: ", output_sparse_depth_dir)

    # collect projection parameters across split files given in data_splits folder
    img_sizes, projective_matrices, R_rect_00s = {}, {}, {}
    extrinsics = {}
    lines = []
    for split_file_name in split_file_names:
        split_file_path = data_split_dir / split_file_name
        assert split_file_path.exists(), split_file_path

        lines += split_file_path.read_text().rsplit()

        used_capture_dates = get_captures_date_from_split_file(split_file_path)

        im_s, p_mats, Rs = get_cam_projection_data(used_capture_dates, kitti_raw_root_dir)
        img_sizes.update(im_s)
        projective_matrices.update(p_mats)
        R_rect_00s.update(Rs)

        extrinsics.update(get_extrinsics_for_used_capture_dates(used_capture_dates, kitti_raw_root_dir))
    lines = sorted(list(set(lines)))

    print("logging processed files in ./log_lines.txt")
    with open('./log_lines.txt', 'w') as log_file:
        log_file.write('\n'.join(lines))

    args = (img_sizes, projective_matrices, R_rect_00s, extrinsics, kitti_raw_root_dir, output_sparse_depth_dir,
            downsample_factor, downsample_indexes)
    f = partial(project_and_save, *args)

    tqdm_lines = tqdm(lines, total=len(lines))

    with Parallel(n_jobs=32, backend="loky") as parallel:
        o = parallel(delayed(f)(line) for line in tqdm_lines)

    o = [str(x) for x in o if x is not None]

    if len(o) > 0:
        lines = '\n'.join(o)
        print("logging files where preprocess failed in ./failed_downsamples_log.txt")
        with open('./failed_downsamples_log.txt', 'w') as log_file:
            log_file.write(lines)

    print('Preprocessing of LiDAR data Finished.')

if __name__ == '__main__':
    main()
