from argoverse.utils.camera_stats import (
    CAMERA_LIST,
    RING_CAMERA_LIST,
    STEREO_CAMERA_LIST
)

def validate_camera_option(cameras):
    """
    Just checks the list of cameras names is valid and return a list without duplicates
    Assumes `cameras` is a comma separated list of camera names
    """
    camera_set = set()

    if isinstance(cameras, str):
        cameras = cameras.split(',')

    for camera in cameras:

        if camera == "ring":
            camera_set.update(set(RING_CAMERA_LIST)) # inplace union
        elif camera == "stereo":
            camera_set.update(set(STEREO_CAMERA_LIST))
        elif camera in CAMERA_LIST:
            camera_set.add(camera)
        else:
            raise ValueError(f"Camera of name {camera} is not valid. Cameras available: {CAMERA_LIST}")

    return list(camera_set)