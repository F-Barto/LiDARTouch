"""
Conveniant script to produce lidar-projected images from argoverse tracking dataset

Assumes that argoverse-api is installed: https://github.com/argoai/argoverse-api
"""

import click
from pathlib import Path
import numpy as np
import json
from tqdm import  tqdm
from numpy import savez_compressed
import copy
import os
import pyntcloud
from scipy.spatial.transform import Rotation
import pickle

# Parallel
from joblib import Parallel, delayed
from functools import partial


from argoverse.utils.camera_stats import (
    CAMERA_LIST,
    RING_CAMERA_LIST,
    STEREO_CAMERA_LIST,
    get_image_dims_for_camera
)

from argoverse.utils.calibration import (
    point_cloud_to_homogeneous,
    project_lidar_to_img_motion_compensated
)

vlp32_planes = {31: -25.,
                30: -15.639,
                29: -11.31,
                28: -8.843,
                27: -7.254,
                26: -6.148,
                25: -5.333,
                24: -4.667,
                23: -4.,
                22: -3.667,
                21: -3.333,
                20: -3.,
                19: -2.667,
                18: -2.333,
                17: -2.,
                16: -1.667,
                15: -1.333,
                14: -1.,
                13: -0.667,
                12: -0.333,
                11: 0.,
                10: 0.333,
                9:  0.667,
                8:  1.,
                7:  1.333,
                6:  1.667,
                5:  2.333,
                4:  3.333,
                3:  4.667,
                2:  7.,
                1:  10.333,
                0:  15.}

# copied from https://github.com/irenecortes/argoverse_lidar
def load_ply_ring(ply_fpath) -> np.ndarray:
    """Load a point cloud file from a filepath.

    Args:
        ply_fpath: Path to a PLY file

    Returns:
        arr: Array of shape (N, 3)5
    """

    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]
    i = np.array(data.points.intensity)[:, np.newaxis]
    ring = np.array(data.points.laser_number)[:, np.newaxis]

    return np.concatenate((x, y, z, i, ring), axis=1)


def project_pts_on_cam(pts, calib, camera_name, cam_timestamp, lidar_timestamp, split_dir, log):
    points_h = point_cloud_to_homogeneous(pts).T

    uv, uv_cam, valid_pts_bool = project_lidar_to_img_motion_compensated(
        points_h,  # these are recorded at lidar_time
        copy.deepcopy(calib),
        camera_name,
        int(cam_timestamp),
        int(lidar_timestamp),
        str(split_dir),
        log,
    )

    uv = uv[valid_pts_bool].astype(np.int32)
    uv_cam = uv_cam.T[valid_pts_bool]

    img_width, img_height = get_image_dims_for_camera(camera_name)
    projected_lidar = np.zeros((img_height, img_width))

    projected_lidar[uv[:, 1], uv[:, 0]] = uv_cam[:, 2]

    return projected_lidar

# adapted from https://github.com/irenecortes/argoverse_lidar
def separate_pc(pc, tf_up, tf_down, th=1.0):
    pc_points = np.ones((len(pc), 4))
    pc_points[:, 0:3] = pc[:, 0:3]

    pc_up_tf = np.dot(np.linalg.inv(tf_up), pc_points.transpose())
    pc_down_tf = np.dot(np.linalg.inv(tf_down), pc_points.transpose())

    pc_up_dis = np.sqrt(pc_up_tf[0, :] ** 2 + pc_up_tf[1, :] ** 2 + pc_up_tf[2, :] ** 2)
    pc_up_omega = np.arcsin(pc_up_tf[2, :] / pc_up_dis) * 180 / np.pi

    pc_down_dis = np.sqrt(pc_down_tf[0, :] ** 2 + pc_down_tf[1, :] ** 2 + pc_down_tf[2, :] ** 2)
    pc_down_omega = np.arcsin(pc_down_tf[2, :] / pc_down_dis) * 180 / np.pi

    pc_angles = np.array([vlp32_planes[pc[i, 4]] for i in range(0, len(pc))])

    up_is_min = np.fabs(pc_up_omega - pc_angles) < np.fabs(pc_down_omega - pc_angles)
    down_is_min = ~up_is_min

    up_mask = up_is_min & (np.fabs(pc_up_omega - pc_angles) < th)
    down_mask = down_is_min & (np.fabs(pc_down_omega - pc_angles) < th)

    pc_up_xyz = pc_points.T[:, up_mask]
    pc_down_xyz = pc_points.T[:, down_mask]

    pc_up = np.zeros((pc_up_xyz.shape[1], 5))
    pc_down = np.zeros((pc_down_xyz.shape[1], 5))

    pc_up[:, 0:3] = pc_up_xyz.transpose()[:, 0:3]
    pc_down[:, 0:3] = pc_down_xyz.transpose()[:, 0:3]
    pc_up[:, 3:5] = pc[up_mask, 3:5]
    pc_down[:, 3:5] = pc[down_mask, 3:5]

    return pc_up, pc_down


def project_and_save(argoverse_tracking_root_dir, camera_list, output_base_dir, beams,
                     pc_part_to_keep, sample_paths):

    relative_lidar_path = Path(sample_paths[0])

    split = relative_lidar_path.parents[2].stem
    log = relative_lidar_path.parents[1].stem
    lidar_timestamp = int(relative_lidar_path.stem[3:])

    lidar_filepath = argoverse_tracking_root_dir / relative_lidar_path

    split_dir = argoverse_tracking_root_dir / split

    log_dir = split_dir / log
    with open(str(log_dir / "vehicle_calibration_info.json"), "r") as f:
        calib_data = json.load(f)


    pc = load_ply_ring(str(lidar_filepath))

    if pc_part_to_keep is not None:

        tf_down_lidar_rot = Rotation.from_quat(calib_data['vehicle_SE3_down_lidar_']['rotation']['coefficients'])
        tf_down_lidar_tr = calib_data['vehicle_SE3_down_lidar_']['translation']
        tf_down_lidar = np.eye(4)
        tf_down_lidar[0:3, 0:3] = tf_down_lidar_rot.as_matrix()
        tf_down_lidar[0:3, 3] = tf_down_lidar_tr

        tf_up_lidar_rot = Rotation.from_quat(calib_data['vehicle_SE3_up_lidar_']['rotation']['coefficients'])
        tf_up_lidar_tr = calib_data['vehicle_SE3_up_lidar_']['translation']
        tf_up_lidar = np.eye(4)
        tf_up_lidar[0:3, 0:3] = tf_up_lidar_rot.as_matrix()
        tf_up_lidar[0:3, 3] = tf_up_lidar_tr

        pc_up, pc_down = separate_pc(pc, tf_up_lidar, tf_down_lidar, th=1.0)

        if pc_part_to_keep == 'up':
            pc = pc_up
            del pc_down
        else: # down
            pc = pc_down
            del pc_up

    if beams is not None:
        mask = np.logical_or.reduce([pc[:, 4] == beam for beam in beams])
        pts = pc[mask][:, :3]
    else:
        pts = pc[:, :3]

    points_h = point_cloud_to_homogeneous(pts).T

    for cam_idx, camera_name in enumerate(camera_list):

        img_rel_path = sample_paths[1 + cam_idx][0]
        closest_cam_timestamp = int(Path(img_rel_path).stem[len(camera_name) + 1:])

        uv, uv_cam, valid_pts_bool = project_lidar_to_img_motion_compensated(
            points_h,  # these are recorded at lidar_time
            copy.deepcopy(calib_data),
            camera_name,
            closest_cam_timestamp,
            lidar_timestamp,
            str(split_dir),
            log,
        )

        img_width, img_height = get_image_dims_for_camera(camera_name)
        img = np.zeros((img_height, img_width))

        if valid_pts_bool is None or uv is None:
            print(f"uv or valid_pts_bool is None: camera {camera_name} ts {closest_cam_timestamp}, {lidar_filepath}")
        else:
            uv = uv[valid_pts_bool].astype(np.int32)  # Nx2 projected coords in pixels
            uv_cam = uv_cam.T[valid_pts_bool]  # Nx3 projected coords in meters

            img[uv[:, 1], uv[:, 0]] = uv_cam[:, 2]  # image of projected lidar measurements

        lidar_filename = lidar_filepath.stem
        output_dir = output_base_dir / split / log / camera_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (lidar_filename + ".npz")
        savez_compressed(str(output_path), img)

    return None


@click.command()
@click.argument('argoverse_tracking_root_dir', type=click.Path(exists=True, file_okay=False)) # .../argoverse-tracking/ under which you can find "train1", "train2", ...
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.argument('data_pkl', type=click.Path(exists=True, file_okay=True))
@click.option('--beams', type=str)
@click.option('--separate_pc', type=click.Choice(['up', 'down']))
def main(argoverse_tracking_root_dir, output_dir, data_pkl, beams, separate_pc):
    print('Preprocessing data....')
    print("INPUT DIR: ", argoverse_tracking_root_dir)
    print("DATA PKL: ", data_pkl)
    print("OUTPUT DIR: ", output_dir)

    argoverse_tracking_root_dir = Path(argoverse_tracking_root_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    with open(data_pkl, 'rb') as f:
        data = pickle.load(f)

    beams_suffix = "all_beams"
    if beams is not None:
        beams = beams.split(',')
        beams_suffix = 'beams_' + '_'.join(beams)
        beams = [int(b) for b in beams]
        print('beams:', beams)

    separate_pc_suffix = "up_and_down"
    if separate_pc is not None:
        separate_pc_suffix = separate_pc
        print('separate_pc: ', separate_pc)

    output_base_dir = output_dir / f"{separate_pc_suffix}_{beams_suffix}"

    for split_name, split_data in data.items():

        camera_list = split_data['camera_list']
        print(f"split {split_name} | camera_list: {camera_list}")
        samples_paths = split_data['samples_paths']

        args = (argoverse_tracking_root_dir, camera_list, output_base_dir, beams, separate_pc)
        f = partial(project_and_save, *args)

        tqdm_samples_paths = tqdm(samples_paths, desc=f"{split_name}", total=len(samples_paths))

        with Parallel(n_jobs=8, prefer="threads") as parallel:
            parallel(delayed(f)(sample) for sample in tqdm_samples_paths)

    print('Preprocessing of LiDAR data Finished.')

if __name__ == '__main__':
    main()