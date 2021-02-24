import torch

def euler2mat(angle):
    """Convert euler angles to rotation matrix"""
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz, cosz, zeros,
                        zeros, zeros, ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny,
                        zeros, ones, zeros,
                        -siny, zeros, cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros, cosx, -sinx,
                        zeros, sinx, cosx], dim=1).view(B, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat

def pose_vec2mat(vec, mode='euler'):
    """Convert Euler parameters to transformation matrix."""
    if mode is None:
        return vec
    trans, rot = vec[:, :3].unsqueeze(-1), vec[:, 3:]
    if mode == 'euler':
        rot_mat = euler2mat(rot)
    else:
        raise ValueError('Rotation mode not supported {}'.format(mode))
    mat = torch.cat([rot_mat, trans], dim=2)  # [B,3,4]
    return mat

def invert_pose(T):
    """Inverts a [B,4,4] torch.tensor pose"""
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).repeat([len(T), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv

class Pose:
    """
    Pose class, that encapsulates a [4,4] transformation matrix
    for a specific reference frame
    """
    def __init__(self, mat):
        """
        Initializes a Pose object.
        Parameters
        ----------
        mat : torch.Tensor [B,4,4]
            Transformation matrix
        """
        assert tuple(mat.shape[-2:]) == (4, 4)
        if mat.dim() == 2:
            mat = mat.unsqueeze(0)
        assert mat.dim() == 3
        self.mat = mat

    def __len__(self):
        """Batch size of the transformation matrix"""
        return len(self.mat)

    @classmethod
    def identity(cls, N=1, device=None, dtype=torch.float):
        """Initializes as a [4,4] identity matrix"""
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([N,1,1]))

    @classmethod
    def from_vec(cls, vec, mode):
        """Initializes from a [B,6] batch vector"""
        mat = pose_vec2mat(vec, mode)  # [B,3,4]
        pose = torch.eye(4, device=vec.device, dtype=vec.dtype).repeat([len(vec), 1, 1])
        pose[:, :3, :3] = mat[:, :3, :3]
        pose[:, :3, -1] = mat[:, :3, -1]
        return cls(pose)

    @property
    def shape(self):
        """Returns the transformation matrix shape"""
        return self.mat.shape

    @property
    def translation(self):
        """Returns the translation component"""
        return self.mat[:, :3, -1]

    @translation.setter
    def translation(self, translation):
        """set the translation component"""
        self.mat[:, :3, -1] = translation

    @property
    def rotation(self):
        """Returns the translation component"""
        return self.mat[:, :3, :3]

    @rotation.setter
    def rotation(self, rotation):
        """set the translation component"""
        self.mat[:, :3, :3] = rotation

    def item(self):
        """Returns the transformation matrix"""
        return self.mat

    def repeat(self, *args, **kwargs):
        """Repeats the transformation matrix multiple times"""
        self.mat = self.mat.repeat(*args, **kwargs)
        return self

    def inverse(self):
        """Returns a new Pose that is the inverse of this one"""
        return Pose(invert_pose(self.mat))

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.mat = self.mat.to(*args, **kwargs)
        return self

    def transform_pose(self, pose):
        """Creates a new pose object that compounds this and another one (self * pose)"""
        assert tuple(pose.shape[-2:]) == (4, 4)
        return Pose(self.mat.bmm(pose.item()))

    def transform_points(self, points):
        """Transforms 3D points using this object"""
        assert points.shape[1] == 3
        B, _, H, W = points.shape
        out = self.mat[:,:3,:3].bmm(points.view(B, 3, -1)) + \
              self.mat[:,:3,-1].unsqueeze(-1)
        return out.view(B, 3, H, W)

    def __matmul__(self, other):
        """Transforms the input (Pose or 3D points) using this object"""
        if isinstance(other, Pose):
            return self.transform_pose(other)
        elif isinstance(other, torch.Tensor):
            if other.shape[1] == 3 and other.dim() > 2:
                assert other.dim() == 3 or other.dim() == 4
                return self.transform_points(other)
            else:
                raise ValueError('Unknown tensor dimensions {}'.format(other.shape))
        else:
            raise NotImplementedError()