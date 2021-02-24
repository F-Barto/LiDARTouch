"""
Conveniant script to produce lidar-projected images from argoverse tracking dataset

Assumes that argoverse-api is installed: https://github.com/argoai/argoverse-api
"""

import click
from pathlib import Path
import numpy as np
from tqdm import  tqdm
from numpy import savez_compressed
import pickle
import copy
import collections
import cv2

from argoverse.utils.camera_stats import CAMERA_LIST, get_image_dims_for_camera
from argoverse.utils.calibration import project_lidar_to_img_motion_compensated, point_cloud_to_homogeneous
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.ply_loader import load_ply

# Parallel
from joblib import Parallel, delayed
from functools import partial


calibrations = {}

###################### IP BASIC #########################################

# code from https://github.com/kujason/ip_basic/blob/master/ip_basic/depth_map_utils.py

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_multiscale(depth_map, max_depth=100.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.
    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict
    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = (depths_in > 30.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.1)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.1)
    valid_pixels_med = (dilated_med > 0.1)
    valid_pixels_far = (dilated_far > 0.1)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.1)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.1)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict

###################### LIDAR Projection ###########################


def get_neighbouring_lidar_timestamps(db, log, initial_timestamp, neighbours_indexes):
    """
    :param db: argoverse SynchronizationDB object
    :param log: string, log name
    :param initial_timestamp: the timestamp from which we want the neighbouring lidar timestamps
    :param neighbours_indexes: a list of neighbouring frames indexes (int), e.g., [-1,0,1]
    :return:
    """

    if 0 not in neighbours_indexes:
        neighbours_indexes.append(0)

    neighbours_indexes = list(set(neighbours_indexes)) # discard duplicates
    neighbours_indexes.sort()

    lidar_timestamps = db.per_log_lidartimestamps_index[log]
    timestamp_idx = np.searchsorted(lidar_timestamps, initial_timestamp)

    neighbours_timestamps = []
    for neighbour_index in neighbours_indexes:
        if timestamp_idx + neighbour_index < 0 or timestamp_idx + neighbour_index >= len(lidar_timestamps):
            return None
        neighbour_timestamp = lidar_timestamps[timestamp_idx + neighbour_index]
        neighbours_timestamps.append(neighbour_timestamp)

    return neighbours_timestamps

def project_and_save(argo_tracking_root_dir, output_base_dir, camera_list,
                     db, calibs, acc_sweeps, ip_basic, samples_paths):

    lidar_filepath = Path(argo_tracking_root_dir) / samples_paths[0]

    split = lidar_filepath.parents[2].stem
    split_dir = argo_tracking_root_dir / split
    log = lidar_filepath.parents[1].stem
    lidar_timestamp = lidar_filepath.stem[3:]

    lidar_timestamps = get_neighbouring_lidar_timestamps(db, log, lidar_timestamp, acc_sweeps)

    if lidar_timestamps is None:
        print(f"Lidar frame of timestamp {lidar_timestamp} from log {log} is at boundary. Not processing it.")
        return None

    for i, camera_name in enumerate(camera_list):
        img_cam_path = samples_paths[i+1][0]
        cam_timestamp = Path(img_cam_path).stem[len(camera_name) + 1:]

        uvs, uv_cams = None, None

        for ts in lidar_timestamps:
            current_lidar_path = lidar_filepath.parent / f"PC_{ts}.ply"

            curr_pts = load_ply(str(current_lidar_path)) # point cloud, numpy array Nx3 -> N 3D coords
            curr_points_h = point_cloud_to_homogeneous(curr_pts).T

            curr_uv, curr_uv_cam, curr_valid_pts_bool = project_lidar_to_img_motion_compensated(
                curr_points_h,  # these are recorded at lidar_time
                copy.deepcopy(calibs[log]),
                camera_name,
                int(cam_timestamp),
                int(ts),
                str(split_dir),
                log,
            )

            curr_uv = curr_uv[curr_valid_pts_bool].astype(np.int32)
            curr_uv_cam = curr_uv_cam.T[curr_valid_pts_bool]

            uvs = curr_uv if uvs is None else np.concatenate([uvs, curr_uv])
            uv_cams = curr_uv_cam if uv_cams is None else np.concatenate([uv_cams, curr_uv_cam])

        img_width, img_height = get_image_dims_for_camera(camera_name)
        acc_projected_lidar = np.zeros((img_height, img_width))
        acc_projected_lidar[uvs[:, 1], uvs[:, 0]] = uv_cams[:, 2]  # image of projected lidar measurements

        if ip_basic:
            acc_projected_lidar = fill_in_multiscale(acc_projected_lidar, max_depth=120.0,
                                                     dilation_kernel_far=CROSS_KERNEL_3,
                                                     dilation_kernel_med=CROSS_KERNEL_5,
                                                     dilation_kernel_near=CROSS_KERNEL_7,
                                                     extrapolate=False,
                                                     blur_type='bilateral',
                                                     show_process=False)[0]

        lidar_filename = lidar_filepath.stem
        output_dir = output_base_dir / log / camera_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (lidar_filename + ".npz")
        savez_compressed(str(output_path), acc_projected_lidar)

    return None

@click.command()
@click.argument('argo_tracking_root_dir', type=click.Path(exists=True, file_okay=False)) # .../argoverse-tracking/ under which you can find "train1", "train2", ...
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.argument('data_pkl', type=click.Path(exists=True, file_okay=True))
@click.option('--acc_sweeps', type=int)
@click.option('--ip_basic', is_flag=True)
def main(argo_tracking_root_dir, output_dir, data_pkl, acc_sweeps, ip_basic):
    print('Preprocessing data....')
    print("INPUT DIR: ", argo_tracking_root_dir)
    print("DATA PKL: ", data_pkl)
    print("OUTPUT DIR: ", output_dir)

    with open(data_pkl, 'rb') as f:
        data = pickle.load(f)

    acc_sweeps = list(range(-acc_sweeps//2,acc_sweeps//2))
    argo_tracking_root_dir = Path(argo_tracking_root_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    for split_name, split_data in data.items():

        camera_list = split_data['camera_list']
        calibs = split_data['calibs']
        samples_paths = split_data['samples_paths']

        split_dir = argo_tracking_root_dir / split_name

        db = SynchronizationDB(str(split_dir))

        output_base_dir = output_dir / split_name

        args = (argo_tracking_root_dir, output_base_dir, camera_list, db, calibs, acc_sweeps, ip_basic)
        f = partial(project_and_save, *args)

        tqdm_samples_paths = tqdm(samples_paths, desc=f"{split_dir.stem}", total=len(samples_paths))

        with Parallel(n_jobs=16, prefer="threads") as parallel:
            parallel(delayed(f)(sample) for sample in tqdm_samples_paths)




    print('Preprocessing of LiDAR data Finished.')

if __name__ == '__main__':
    main()