"""Provides helper methods for loading and parsing KITTI Oxts data.

adapted from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/datasets/kitti_dataset.py
and https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/datasets/kitti_dataset_utils.py#L138
"""

from collections import namedtuple
import numpy as np
import os


# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')

CALIB_FILE = {
    'cam2cam': 'calib_cam_to_cam.txt',
    'velo2cam': 'calib_velo_to_cam.txt',
    'imu2velo': 'calib_imu_to_velo.txt',
}

IMAGE_FOLDER = {
    'left': 'image_02',
    'right': 'image_03',
}

OXTS_POSE_DATA = 'oxts'


def rotx(t):
    """
    Rotation about the x-axis
    Parameters
    ----------
    t : float
        Theta angle
    Returns
    -------
    matrix : np.array [3,3]
        Rotation matrix
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """
    Rotation about the y-axis
    Parameters
    ----------
    t : float
        Theta angle
    Returns
    -------
    matrix : np.array [3,3]
        Rotation matrix
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """
    Rotation about the z-axis
    Parameters
    ----------
    t : float
        Theta angle
    Returns
    -------
    matrix : np.array [3,3]
        Rotation matrix
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def transform_from_rot_trans(R, t):
    """
    Transformation matrix from rotation matrix and translation vector.
    Parameters
    ----------
    R : np.array [3,3]
        Rotation matrix
    t : np.array [3]
        translation vector
    Returns
    -------
    matrix : np.array [4,4]
        Transformation matrix
    """
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary
    Parameters
    ----------
    filepath : str
        File path to read from
    Returns
    -------
    calib : dict
        Dictionary with calibration values
    """
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def pose_from_oxts_packet(raw_data, scale):
    """
    Helper method to compute a SE(3) pose matrix from an OXTS packet
    Parameters
    ----------
    raw_data : dict
        Oxts data to read from
    scale : float
        Oxts scale
    Returns
    -------
    R : np.array [3,3]
        Rotation matrix
    t : np.array [3]
        Translation vector
    """
    packet = OxtsPacket(*raw_data)
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t


def load_oxts_packets_and_poses(oxts_files):
    """
    Generator to read OXTS ground truth data.
    Poses are given in an East-North-Up coordinate system
    whose origin is the first GPS position.
    Parameters
    ----------
    oxts_files : list of str
        List of oxts files to read from
    Returns
    -------
    oxts : list of dict
        List of oxts ground-truth data
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                T_w_imu = transform_from_rot_trans(R, t - origin)

                oxts.append(OxtsData(packet, T_w_imu))

    return oxts


class KITTIRawOxts():

    def __init__(self):
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        return os.path.abspath(os.path.join(image_file, "../../../.."))

    def _get_imu2cam_transform(self, image_file):
        """Gets the transformation between IMU an camera from an image file"""
        parent_folder = self._get_parent_folder(image_file)
        if image_file in self.imu2velo_calib_cache:
            return self.imu2velo_calib_cache[image_file]

        cam2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['cam2cam']))
        imu2velo = read_calib_file(os.path.join(parent_folder, CALIB_FILE['imu2velo']))
        velo2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['velo2cam']))

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
        self.imu2velo_calib_cache[image_file] = imu2cam
        return imu2cam

    @staticmethod
    def _get_oxts_file(image_file):
        """Gets the oxts file from an image file."""
        # find oxts pose file
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return file name
            if IMAGE_FOLDER[cam] in image_file:
                return image_file.replace(IMAGE_FOLDER[cam], OXTS_POSE_DATA).replace('.png', '.txt')
        # Something went wrong (invalid image file)
        raise ValueError('Invalid KITTI path for pose supervision.')

    def _get_oxts_data(self, image_file):
        """Gets the oxts data from an image file."""
        oxts_file = self._get_oxts_file(image_file)
        if oxts_file in self.oxts_cache:
            oxts_data = self.oxts_cache[oxts_file]
        else:
            oxts_data = np.loadtxt(oxts_file, delimiter=' ', skiprows=0)
            self.oxts_cache[oxts_file] = oxts_data
        return oxts_data

    def _get_pose(self, image_file):
        """Gets the pose information from an image file."""
        image_file = str(image_file)

        if image_file in self.pose_cache:
            return self.pose_cache[image_file]
        # Find origin frame in this sequence to determine scale & origin translation
        base, ext = os.path.splitext(os.path.basename(image_file))
        origin_frame = os.path.join(os.path.dirname(image_file), str(0).zfill(len(base)) + ext)
        # Get origin data
        origin_oxts_data = self._get_oxts_data(origin_frame)
        lat = origin_oxts_data[0]
        scale = np.cos(lat * np.pi / 180.)
        # Get origin pose
        origin_R, origin_t = pose_from_oxts_packet(origin_oxts_data, scale)
        origin_pose = transform_from_rot_trans(origin_R, origin_t)
        # Compute current pose
        oxts_data = self._get_oxts_data(image_file)
        R, t = pose_from_oxts_packet(oxts_data, scale)
        pose = transform_from_rot_trans(R, t)
        # Compute odometry pose
        imu2cam = self._get_imu2cam_transform(image_file)
        odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose @ np.linalg.inv(imu2cam)).astype(np.float32)
        # Cache and return pose
        self.pose_cache[image_file] = odo_pose
        return odo_pose

    def get_pose_from_path(self, file_path):
        return self._get_pose(file_path)

    @staticmethod
    def invert_pose_numpy(T):
        """Inverts a [4,4] np.array pose"""
        Tinv = np.copy(T)
        R, t = Tinv[:3, :3], Tinv[:3, 3]
        Tinv[:3, :3], Tinv[:3, 3] = R.T, - np.matmul(R.T, t)
        return Tinv

    def get_transformations_between_pairs(self, target_file_path, source_file_paths):
        target_pose = self._get_pose(target_file_path)
        source_poses = [self._get_pose(source_file_path) for source_file_path in source_file_paths]

        # computes the "relative" transformation between the sources and target poses
        relatives_transformations = [self.invert_pose_numpy(source_pose) @ target_pose for source_pose in source_poses]

        return relatives_transformations

    def get_magnitude_between_pairs(self, target_file_path, source_file_paths):

        relatives_transformations = self.get_transformations_between_pairs(target_file_path, source_file_paths)

        magnitudes = [np.linalg.norm(rel_t[:3, 3]) for rel_t in relatives_transformations]

        return np.stack(magnitudes, axis=0)