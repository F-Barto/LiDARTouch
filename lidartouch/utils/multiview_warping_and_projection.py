import torch
import torch.nn.functional as F
from functools import lru_cache

from lidartouch.utils.camera import Camera

@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    """
    Create meshgrid with a specific resolution
    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1
    Returns
    -------
    xs : torch.Tensor [B,1,W]
        Meshgrid in dimension x
    ys : torch.Tensor [B,H,1]
        Meshgrid in dimension y
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs], indexing='ij')

    """ example for H,W = 3,4
    >>> xs
    tensor([[0., 1., 2., 3.],
            [0., 1., 2., 3.],
            [0., 1., 2., 3.]])
    >>> ys
    tensor([[0., 0., 0., 0.],
            [1., 1., 1., 1.],
            [2., 2., 2., 2.]])
    """

    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1]) # just repeat along batch axis; xs.shape = [B, H, W]

@lru_cache(maxsize=None)
def image_grid(B, H, W, dtype, device, normalized=False):
    """
    Create an image grid with a specific resolution
    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1
    Returns
    -------
    grid : torch.Tensor [B,3,H,W]
        Image grid containing a meshgrid in x, y and 1
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=1)

    """ example for B,C,H,W = 2,1,3,4
    >>> grid[0] # first element of batch
    tensor([[[0., 1., 2., 3.],
             [0., 1., 2., 3.],
             [0., 1., 2., 3.]],     # xs
    
            [[0., 0., 0., 0.],
             [1., 1., 1., 1.],
             [2., 2., 2., 2.]],     # ys
    
            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]])    # ones (zs that will be scaled by predicted depth)
    """

    return grid




################################################
################# View Warping #################
################################################

def view_synthesis(ref_image, depth, ref_cam: Camera, cam: Camera, mode='bilinear', padding_mode='zeros',
                   align_corners=True, return_valid_mask=False):
    """
    Synthesize an image from another plus a depth map.
    Parameters
    ----------
    ref_image : torch.Tensor [B,3,H,W]
        Reference image to be warped
    depth : torch.Tensor [B,1,H,W]
        Depth map from the original image
    ref_cam : utils.camera.Camera
        Camera class for the reference image
    cam : utils.camera.Camera
        Camera class for the original image
    mode : str
        Interpolation mode
    padding_mode : str
        Padding mode for interpolation
    Returns
    -------
    ref_warped : torch.Tensor [B,3,H,W]
        Warped reference image in the original frame of reference
    """
    if depth.size(1) != 1:
        raise ValueError(f"depth tensor should be of shape [B,1,H,W]. depth.shape: {depth.shape}")
    # Reconstruct world points from target_camera
    world_points = reconstruct(cam, depth, frame='w')
    # Project world points onto reference camera
    ref_coords, valid_mask = project(ref_cam, world_points, frame='w', return_valid_mask=True)
    # View-synthesis given the projected reference points
    view_synthesized = F.grid_sample(ref_image, ref_coords, mode=mode, padding_mode=padding_mode,
                                     align_corners=align_corners)

    return (view_synthesized, valid_mask) if return_valid_mask else view_synthesized


def project(camera: Camera, X, frame='w', return_valid_mask=False):
    """
    Projects 3D points onto the image plane

    Parameters
    ----------
    camera: utils.camera.Camera object
        Camera object representing a pinhole model with all its intrinsics as attributes
    X : torch.Tensor [B,3,H,W]
        3D points to be projected
    frame : 'w'
        Reference frame: 'c' for camera and 'w' for world
    Returns
    -------
    points : torch.Tensor [B,H,W,2]
        2D projected points that are within the image boundaries
    """
    B, C, H, W = X.shape
    assert C == 3

    # Project 3D points onto the camera image plane
    if frame == 'c':
        Xc = camera.K.bmm(X.view(B, 3, -1))
    elif frame == 'w':
        Xc = camera.K.bmm((camera.Tcw @ X).view(B, 3, -1))
    else:
        raise ValueError('Unknown reference frame {}'.format(frame))

    # Normalize points
    X = Xc[:, 0]
    Y = Xc[:, 1]
    Z = Xc[:, 2].clamp(min=1e-5)
    Xnorm = 2 * (X / Z) / (W - 1) - 1.
    Ynorm = 2 * (Y / Z) / (H - 1) - 1.

    # True for pixel projecting in the image frame (valid picels) False otherwise (invalid pixels)
    Xmask = ((Xnorm < 1) & (Xnorm > -1)).detach()
    Ymask = ((Ynorm < 1) & (Ynorm > -1)).detach()
    valid_mask = Xmask & Ymask

    pixel_coordinates = torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)
    valid_mask = valid_mask.view(B, H, W, 1).permute(0,3,1,2)

    return (pixel_coordinates, valid_mask) if return_valid_mask else pixel_coordinates


def reconstruct(camera: Camera, depth, frame='w'):
        """
        Reconstructs pixel-wise 3D points from a depth map.

        Parameters
        ----------
        camera: utils.camera.Camera object
            Camera object representing a pinhole model with all its intrinsics as attributes
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world
        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """
        B, C, H, W = depth.shape
        assert C == 1

        # Create flat index grid
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False) # [B,3,H,W]
        flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

        """ example for H,W = 5,5
        >>> flat_grid[0]
        tensor([[0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4.],
                [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 4., 4., 4., 4., 4.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        """

        # Estimate the outward rays in the camera frame
        xnorm = (camera.Kinv.bmm(flat_grid)).view(B, 3, H, W)  # output similar to grid but coords are in world coords

        # Scale rays to metric depth
        Xc = xnorm * depth

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            return camera.Twc @ Xc
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))
