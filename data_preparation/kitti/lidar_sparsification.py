import click
import numpy as np
from tqdm import  tqdm
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial

from utils import (get_extrinsics_for_used_capture_dates, get_captures_date_from_split_file, get_cam_projection_data,
                    determine_valid_cam_coords, get_meta, downsample_motion_comp_lidar_pc)

split_file_names = ['depth_completion_train.txt', 'depth_completion_val.txt', 'eigen_test_files.txt',
                    'eigen_train_files.txt', 'eigen_val_files.txt']

'''
python ./lidar_sparsification.py \
/home/clear/fbartocc/data/KITTI_raw/ \
/home/clear/fbartocc/working_data/KITTI/test \
/home/clear/fbartocc/depth_project/Depth/data_splits \
--downsample_indexes='5,7,9,11'
'''

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
    except:
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
@click.option('--downsample_indexes', type=str)
@click.option('--downsample_factor', type=int)
def main(kitti_raw_root_dir, output_dir, data_split_dir, downsample_indexes=None, downsample_factor=None):

    print('Preprocessing data....')
    print("KITTI RAW DIR: ", kitti_raw_root_dir)

    assert (downsample_factor is not None) != (downsample_indexes is not None)
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

        lines += split_file_path.read_text().rsplit()

        used_capture_dates = get_captures_date_from_split_file(split_file_path)

        im_s, p_mats, Rs = get_cam_projection_data(used_capture_dates, kitti_raw_root_dir)
        img_sizes.update(im_s)
        projective_matrices.update(p_mats)
        R_rect_00s.update(Rs)

        extrinsics.update(get_extrinsics_for_used_capture_dates(used_capture_dates, kitti_raw_root_dir))
    lines = list(set(lines))

    args = (img_sizes, projective_matrices, R_rect_00s, extrinsics, kitti_raw_root_dir, output_sparse_depth_dir,
            downsample_indexes, downsample_factor)
    f = partial(project_and_save, *args)

    tqdm_lines = tqdm(lines, total=len(lines))

    with Parallel(n_jobs=32, backend="loky") as parallel:
        o = parallel(delayed(f)(line) for line in tqdm_lines)

    o = [str(x) for x in o if x is not None]

    if len(o) > 0:
        lines = '/n'.join(o)
        with open('./failed_downsamples_log.txt', 'w') as log_file:
            log_file.write(lines)

    print('Preprocessing of LiDAR data Finished.')

if __name__ == '__main__':
    main()