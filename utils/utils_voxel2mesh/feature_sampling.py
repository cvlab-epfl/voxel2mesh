import torch.nn as nn
import torch
from itertools import product
import torch.nn.functional as F
from utils.utils_voxel2mesh.graph_conv import GraphConvEdgeLengthWeighted as GraphConv 
from IPython import embed

class SkipConnections(nn.Module): 

    def __init__(self, config, features_count):
        super(SkipConnections, self).__init__()

        if config.config.low_resolution is None:
            D, H, W = config.config.hint_patch_shape
        else:
            D, H, W = config.config.low_resolution
        # assert D == H == W, 'should be of same dim'
        self.shape = torch.tensor([W, H, D]).cuda().float()

        self.shift = torch.tensor(list(product((-1, 0, 1), repeat=3)))[None].float() * torch.tensor([[[2/(W-1), 2/(H-1), 2/(D-1)]]])[None]
        self.shift = self.shift.cuda()

        self.sum_neighbourhood = nn.Conv2d(features_count, features_count, kernel_size=(1, 27), padding=0).cuda()

        torch.nn.init.kaiming_normal_(self.sum_neighbourhood.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.sum_neighbourhood.bias, 0)

    def forward(self, voxel_features, vertices):
        neighbourhood = vertices[:, :, None, None] + self.shift[:, :, :, None] 
        features = F.grid_sample(voxel_features, neighbourhood, mode='bilinear', padding_mode='border', align_corners=True)
        features = features[:, :, :, :, 0]
        features = self.sum_neighbourhood(features)[:, :, :, 0].transpose(2, 1)

        return features

class NeighbourhoodSampling(nn.Module):

    def __init__(self, config, features_count, step):
        super(NeighbourhoodSampling, self).__init__()

        if config.config.low_resolution is None:
            D, H, W = config.config.hint_patch_shape
        else:
            D, H, W = config.config.low_resolution 
        self.shape = torch.tensor([W, H, D]).cuda().float()

        self.shift = torch.tensor(list(product((-1, 0, 1), repeat=3)))[None].float() * torch.tensor([[[2 ** (config.steps+1 - step)/(W), 2 ** (config.steps+1 - step)/(H), 2 ** (config.steps+1 - step)/(D)]]])[None]
        self.shift = self.shift.cuda()

        self.sum_neighbourhood = nn.Conv2d(features_count, features_count, kernel_size=(1, 27), padding=0).cuda()

        # torch.nn.init.kaiming_normal_(self.sum_neighbourhood.weight, nonlinearity='relu')
        # torch.nn.init.constant_(self.sum_neighbourhood.bias, 0)

        self.feature_diff_1 = nn.Linear(features_count, features_count)
        self.feature_diff_2 = nn.Linear(features_count, features_count) 

        self.feature_center_1 = nn.Linear(features_count, features_count)
        self.feature_center_2 = nn.Linear(features_count, features_count)

    def forward(self, voxel_features, vertices):
        neighbourhood = vertices[:, :, None, None] + self.shift[:, :, :, None] 
        features = F.grid_sample(voxel_features, neighbourhood, mode='bilinear', padding_mode='border', align_corners=True)
        features = features[:, :, :, :, 0]

        features_diff_from_center = features - features[:,:,:,13][:,:,:,None] # 13 is the index of the center cordinate in shifts
        features_diff_from_center = features_diff_from_center.permute([0,3,2,1])
        features_diff_from_center = self.feature_diff_1(features_diff_from_center)
        features_diff_from_center = self.feature_diff_2(features_diff_from_center)
        features_diff_from_center = features_diff_from_center.permute([0,3,2,1])
        
        features_diff_from_center = self.sum_neighbourhood(features_diff_from_center)[:, :, :, 0].transpose(2, 1)

        center_feautres =  features[:,:,:,13].transpose(2, 1)
        center_feautres = self.feature_center_1(center_feautres)
        center_feautres = self.feature_center_2(center_feautres)

        features = center_feautres + features_diff_from_center 
        return features

class LearntNeighbourhoodSampling(nn.Module):

    def __init__(self, config, features_count, step):
        super(LearntNeighbourhoodSampling, self).__init__()

        D, H, W = config.patch_shape 
        self.shape = torch.tensor([W, H, D]).cuda().float()

        self.shift = torch.tensor(list(product((-1, 0, 1), repeat=3)))[None].float() * torch.tensor([[[2 ** (config.steps+1 - step)/(W), 2 ** (config.steps+1 - step)/(H), 2 ** (config.steps+1 - step)/(D)]]])[None]
        self.shift = self.shift.cuda()

        self.sum_neighbourhood = nn.Conv2d(features_count, features_count, kernel_size=(1, 27), padding=0).cuda()

        # torch.nn.init.kaiming_normal_(self.sum_neighbourhood.weight, nonlinearity='relu')
        # torch.nn.init.constant_(self.sum_neighbourhood.bias, 0)
        self.shift_delta = nn.Conv1d(features_count, 27*3, kernel_size=(1), padding=0).cuda()
        self.shift_delta.weight.data.fill_(0.0)
        self.shift_delta.bias.data.fill_(0.0)

        self.feature_diff_1 = nn.Linear(features_count + 3, features_count)
        self.feature_diff_2 = nn.Linear(features_count, features_count) 

        self.feature_center_1 = nn.Linear(features_count + 3, features_count)
        self.feature_center_2 = nn.Linear(features_count, features_count)

    def forward(self, voxel_features, vertices):

        B, N, _ = vertices.shape
        center = vertices[:, :, None, None]
        features = F.grid_sample(voxel_features, center, mode='bilinear', padding_mode='border', align_corners=True)
        features = features[:, :, :, 0, 0]
        shift_delta = self.shift_delta(features).permute(0, 2, 1).view(B, N, 27, 1, 3)
        shift_delta[:,:,0,:,:] = shift_delta[:,:,0,:,:] * 0 # setting first shift to zero so it samples at the exact point
 
        # neighbourhood = vertices[:, :, None, None] + self.shift[:, :, :, None] + shift_delta
        neighbourhood = vertices[:, :, None, None] + shift_delta
        features = F.grid_sample(voxel_features, neighbourhood, mode='bilinear', padding_mode='border', align_corners=True)
        features = features[:, :, :, :, 0]
        features = torch.cat([features, neighbourhood.permute(0,4,1,2,3)[:,:,:,:,0]], dim=1)

        features_diff_from_center = features - features[:,:,:,0][:,:,:,None] # 0 is the index of the center cordinate in shifts
        features_diff_from_center = features_diff_from_center.permute([0,3,2,1])
        features_diff_from_center = self.feature_diff_1(features_diff_from_center)
        features_diff_from_center = self.feature_diff_2(features_diff_from_center)
        features_diff_from_center = features_diff_from_center.permute([0,3,2,1])
        
        features_diff_from_center = self.sum_neighbourhood(features_diff_from_center)[:, :, :, 0].transpose(2, 1)

        center_feautres =  features[:,:,:,13].transpose(2, 1)
        center_feautres = self.feature_center_1(center_feautres)
        center_feautres = self.feature_center_2(center_feautres)

        features = center_feautres + features_diff_from_center 
        return features


class BasicSkipConnection(nn.Module):

    def __init__(self, config, features_count):
        super(BasicSkipConnection, self).__init__()
 

    def forward(self, voxel_features, vertices):
        
        neighbourhood = vertices[:, :, None, None]  
        features = F.grid_sample(voxel_features, neighbourhood, mode='bilinear', padding_mode='border', align_corners=True)
        features = features[:, :, :, 0, 0].transpose(2, 1)

        # features = self.sum_neighbourhood(features)[:, :, :, 0].transpose(2, 1)

        return features



