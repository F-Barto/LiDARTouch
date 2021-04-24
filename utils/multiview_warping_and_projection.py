import torch
import torch.nn.functional as F
from functools import lru_cache

from utils.camera import Camera

import torch_scatter


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
    ys, xs = torch.meshgrid([ys, xs])

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

def view_synthesis(ref_image, depth, ref_cam: Camera, cam: Camera,
                   mode='bilinear', padding_mode='zeros', return_valid_mask=False):
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
    assert depth.size(1) == 1
    # Reconstruct world points from target_camera
    world_points = reconstruct(cam, depth, frame='w')
    # Project world points onto reference camera
    ref_coords, valid_mask = project(ref_cam, world_points, frame='w', return_valid_mask=True)
    # View-synthesis given the projected reference points
    view_synthesized = F.grid_sample(ref_image, ref_coords, mode=mode, padding_mode=padding_mode, align_corners=True)

    return view_synthesized, valid_mask if return_valid_mask else view_synthesized



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

    valid_mask = ((X < W) & (Y < H)).detach()

    Xnorm = 2 * (X / Z) / (W - 1) - 1.
    Ynorm = 2 * (Y / Z) / (H - 1) - 1.

    # Clamp out-of-bounds pixels
    # Xmask = ((Xnorm > 1) + (Xnorm < -1)).detach()
    # Xnorm[Xmask] = 2.
    # Ymask = ((Ynorm > 1) + (Ynorm < -1)).detach()
    # Ynorm[Ymask] = 2.

    pixel_coordinates = torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)
    valid_mask = valid_mask.view(B, 1, H, W)

    return pixel_coordinates, valid_mask if return_valid_mask else pixel_coordinates


def packed_to_padded(pc_packed, pc_batch):
    if not pc_packed.dim() == 2:
        raise ValueError(f"pc_packed can only be 2-dimensional (nb_points x C) but has shape {pc_packed.shape}.")
    N, C = pc_packed.shape
    B = pc_batch.max() + 1 # pc_batch contains the batch idx for correspoing point of pc_packed
    first_idxs = [pc_batch.new_zeros(1)]
    for i in range(1, B):
        mask = (pc_batch == i)
        cumsum = mask.cumsum(dim=0)
        first_idx = (cumsum == 1).max(0).indices.unsqueeze(0)
        first_idxs.append(first_idx)
    last_idx = torch.Tensor([N]).to(pc_batch.device)
    first_idxs = torch.cat(first_idxs + [last_idx]) # append nb points as last idx for diff
    # diff is out[i] = input[i + 1] - input[i]
    lens = first_idxs[1:] - first_idxs[:-1] # diff between first idxs = number of points by batch
    lens = lens.long().tolist()
    max_len = max(lens)

    pc_padded = pc_packed.new_zeros((B, C, max_len), requires_grad=True)
    padded_mask = pc_packed.new_zeros((B, max_len, C))

    first_idxs = first_idxs.long().tolist()
    for i, f in enumerate(first_idxs[:-1]):
        pc_padded[i, :, :lens[i]] = pc_packed[f:f+lens[i],:].T
        padded_mask[i, :lens[i], :] = 1

    padded_mask = padded_mask > 0

    return pc_padded, padded_mask

def project_pc(camera: Camera, pc_features, pc_pos, pc_batch_idx, B, H, W):
    """
    Projects 3D points onto the image plane

    Parameters
    ----------
    camera: utils.camera.Camera object
        Camera object representing a pinhole model with all its intrinsics as attributes
    pc_features : torch.Tensor [N,C]
        features associates to each 3D points to be projected on camera plane
    pc_pos : torch.Tensor [N,3]
        coordinates 3D points in rectified camera frame to be projected on camera plane
    pc_batch_idx : torch.Tensor [N,1]
        batch index associated to each 3D points to be projected on camera plane
    frame : 'w'
        Reference frame: 'c' for camera and 'w' for world
    Returns
    -------
    points : torch.Tensor [B,H,W,2]
        2D projected points that are within the image boundaries
    """
    _, C = pc_features.shape

    # Project 3D points onto the camera image plane

    # padded version
    # pc_pos, pad_mask = packed_to_padded(pc_pos, pc_batch_idx) # return BxCxpadded_dim from NxC
    # uv_map = camera.K.bmm(pc_pos) # 3x(N_total) + pad)
    # uv_map = uv_map.transpose(1, 2)[pad_mask].view(-1, 3) # Nx3

    # packed version
    # with camera.K[pc_batch_idx]
    # pc_pos is a set if 3D coordinates from B points clouds (1 per batch)
    # pc_batch_idx indicates for each 3D points its corresponding point cloud
    # However we have one intrinsics per batch, K is of shape Bx3x3
    # we can't directly do a batched matrix-matrix multiplication
    # with camera.K[pc_batch_idx] we replicate the corresponding intrinsic matrix for each 3D point
    # camera.K[pc_batch_idx] is of shape N_totalx3x3
    # so we can use torch.bmm() efficiently at the cost of memory
    # it is equivalent to do: for each point i  camera.K[pc_batch_idx[i]] @ pc_pos[i] (1 point <-> 1 matrix)
    uv_map = camera.K[pc_batch_idx].bmm(pc_pos.unsqueeze(-1))
    uv_map = uv_map.squeeze()  # Nx3x1 -> Nx3

    uv_map[:, 0:2] /= uv_map[:, 2].unsqueeze(1)  # u, v = x/z, y/z
    uv_map[:, :2] = uv_map[:, :2].int().float() # x and y from real to pixel coords

    # filter out-of-frame coords
    valid_mask = (uv_map[:, 0] < W) & (uv_map[:, 1] < H)
    uv_map = uv_map[valid_mask] # N x 3

    X = uv_map[:, 0]
    Y = uv_map[:, 1]
    Z = uv_map[:, 2]

    # 3D-to-1D indexes  ->  Flat[x + HEIGHT * (y + DEPTH * z)] = Original[x, y, z]
    flat_idxs = X.unsqueeze(-1) + H * (Y.unsqueeze(-1) + B * pc_batch_idx)  # N x 1

    # maps each flat index to a unique index in [0, N-1]
    _, inverse_idxs = flat_idxs[:,0].unique(return_inverse=True)

    # see https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/min.html
    _, argmins = torch_scatter.scatter_max(Z, inverse_idxs)

    filtered_uv_map = uv_map[argmins].long()
    filtered_batch_idx = pc_batch_idx[argmins].long()
    filtered_features = pc_features[argmins]

    # project features on image
    uv = pc_pos.new_zeros(B, C, H, W)  # same device and type as pc_pos
    uv[filtered_batch_idx, :, filtered_uv_map[:, 1], filtered_uv_map[:, 0]] = filtered_features

    return uv, filtered_uv_map, filtered_batch_idx

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
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [0,:,15,23] == []
        flat_grid = grid.view(B, 3, -1)  # [B,3,HW]

        """ example for H,W = 5,5
        >>> flat_grid[0]
        tensor([[0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4.],
                [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 4., 4., 4., 4., 4.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        """

        # Estimate the outward rays in the camera frame
        xnorm = (camera.Kinv.bmm(flat_grid)).view(B, 3, H, W) # output similar to grid but coords are in world coords

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