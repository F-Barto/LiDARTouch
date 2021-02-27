import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Conv2d

from torch_geometric.nn import PointConv, fps, radius, global_sort_pool, DeepGraphInfomax, knn_interpolate
from torch_scatter import scatter_add, scatter_max

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(
            Lin(channels[i - 1], channels[i]),
            ReLU(),
            BN(channels[i])
        )
        for i in range(1, len(channels))
    ])

class SAModuleFullPoint(torch.nn.Module):
    def __init__(self, r, sample_size, nn):
        super(SAModuleFullPoint, self).__init__()
        self.r = r
        self.sample_size = sample_size
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        row, col = radius(
            pos, pos, self.r, batch, batch, max_num_neighbors=self.sample_size)
        edge_index = torch.stack([col, row], dim=0)

        x = self.conv(x, (pos, pos), edge_index)
        pos, batch = pos, batch
        return x, pos, batch


class SAModuleMRG(torch.nn.Module):
    def __init__(self, start_point, sampled_point_ammount, layers):
        super(SAModuleMRG, self).__init__()

        self.upsample_needed = sampled_point_ammount < start_point
        self.downsample = DownsampleMRG(sampled_point_ammount / start_point)
        self.layers = torch.nn.ModuleList()
        if(self.upsample_needed):
            self.upsample = UpsampleMRG(start_point, sampled_point_ammount)

        for i in range(len(layers)):
            self.layers.append(layers[i])


    def forward(self, x, pos, batch):

        out_lr = self.downsample(x, pos, batch)

        for i in range(len(self.layers)):
            out_lr = self.layers[i](*out_lr)

        x_lr, pos_lr, batch_lr = out_lr

        if(self.upsample_needed):
            x, pos, batch = self.upsample(x, pos, batch, x_lr, pos_lr, batch_lr)
        else:
            x = x_lr, pos = pos_lr, batch = batch_lr

        return x, pos, batch

class DownsampleMRG(torch.nn.Module):
    def __init__(self, scale_factor):
        assert (scale_factor <= 1)
        assert (scale_factor > 0)

        super(DownsampleMRG, self).__init__()

        self.scale_factor = scale_factor

    def forward(self, x, pos, batch):

        if( self.scale_factor < 1 ):
            downsampled_idx = fps(pos, batch, self.scale_factor)
            x = None if x is None else x[downsampled_idx]
            pos = pos[downsampled_idx]
            batch = batch[downsampled_idx]

        return x, pos, batch

class UpsampleMRG(torch.nn.Module):
    def __init__(self, high_res_points, low_res_points):
        assert (high_res_points > low_res_points)

        super(UpsampleMRG, self).__init__()

        self.high_res_points = high_res_points
        self.low_res_points = low_res_points

    def forward(self, x_hr, pos_hr, batch_hr, x_lr, pos_lr, batch_lr):
        out_x = x_hr
        out_pos = pos_hr
        out_batch = batch_hr
        if(out_x is not None):
            out_x = torch.cat([out_x, out_pos], dim=1)
        else:
            out_x = out_pos

        lr_x = x_lr
        lr_pos = pos_lr
        lr_batch = batch_lr
        lr_x = torch.cat([lr_x, lr_pos], dim=1)

        #return a tensor where the feature of each point of lr_x are appended in a new array in
        # the point of the upsampled version is in his knn
        lr_x = knn_interpolate(lr_x, lr_pos, out_pos, lr_batch, out_batch, k=1)
        out_x = torch.cat([out_x, lr_x], dim=1)

        return out_x, out_pos.new_zeros((out_x.size(0), 3)), out_batch

class GlobalSAModule(torch.nn.Module):
    '''
    One group with all input points, can be viewed as a simple PointNet module.
    It also return the only one output point(set as origin point).
    '''
    def __init__(self, nn=None):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        if self.nn is not None:
            x = self.nn(torch.cat([x, pos], dim=1))
        x = scatter_max(x, batch, dim=0)[0]

        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch



class PointNet2MRGExtractor(torch.nn.Module):
    def __init__(self, out_chans, nfeatures=5, num_points=1024, head='local'):
        super().__init__()

        assert head in ['local', 'global', 'local+global']
        self.local_head = None
        self.global_head = None

        # nfeatures = x dim + pos dim (e.g. nfeatures = 5 -> x dim = 2)
        # (256 + nfeatures + 3)*3 + 3
        # SAModuleMRG + Upsample  -> (256 + nfeatures + 3)=(mlp, x+pos, lr_pos)
        # *3 because repat for high, mid and low res
        # + 3 -> concat pos
        in_chans = (256 + nfeatures + 3)*3 + 3

        if 'local' in head:
            self.local_head = LocalHeadExtractor(in_chans, out_chans)
        if 'global' in head:
            self.global_head = GlobalHeadExtractor(in_chans, out_chans)

        nFeaturesL2 = 3 + 128

        shared_mpls = [
            SAModuleFullPoint(0.4, 16, MLP([nfeatures, 64, 64, 128])),
            SAModuleFullPoint(0.9, 32, MLP([nFeaturesL2, 128, 128, 256]))
        ]

        # The mpls are shared to lower the model memory footprint
        self.high_resolution_module = SAModuleMRG(num_points, 512, shared_mpls)
        self.mid_resolution_module = SAModuleMRG(num_points, 256, shared_mpls)
        self.low_resolution_module = SAModuleMRG(num_points, 128, shared_mpls)


    def forward(self, features, pos, batch_idx):
        sa_out = (features, pos, batch_idx)

        hr_x, hr_pos, hr_batch = self.high_resolution_module(*sa_out)
        mr_x, mr_pos, mr_batch = self.mid_resolution_module(*sa_out)
        x = torch.cat([hr_x, mr_x], dim=1)

        lr_x, lr_pos, lr_batch = self.low_resolution_module(*sa_out)
        x = torch.cat([x, lr_x], dim=1)

        # from there x is the concat of high_res mid_res and low_res features

        if self.global_head is not None and self.local_head is not None:
            return self.global_head(x, pos, batch_idx), self.local_head(x, pos, batch_idx)
        if self.local_head is not None:
            return self.local_head(x, pos, batch_idx)
        if self.global_head is not None:
            return self.global_head(x, pos, batch_idx)

        return x

class LocalHeadExtractor(torch.nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()

        self.last_mlp = MLP([in_chans, 1024, 1024, out_chans])

    def forward(self, x, pos, batch_idx):
        x = torch.cat([x, pos], dim=1)
        x = self.last_mlp(x)

        return x


class GlobalHeadExtractor(torch.nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()

        self.readout = GlobalSAModule(MLP([in_chans, 1024, 1024, 1024]))

        # Classification Layers
        self.lin1 = Lin(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = Lin(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.lin3 = Lin(256, out_chans)
        self.bn3 = nn.BatchNorm1d(out_chans)

    def forward(self, x, pos, batch_idx):
        x, pos, batch = self.readout(x, pos, batch_idx)

        x = F.relu(self.bn1(self.lin1(x)))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.bn3(self.lin3(x)))

        return x