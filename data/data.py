from utils.utils_common import DataModes, crop
from utils import stns
from skimage import measure
 
import torch
import torch.nn.functional as F
# from utils import stns
# from utils.utils_mesh import voxel2mesh, clean_border_pixels
import numpy as np 
from IPython import embed 
 

class Sample:
    def __init__(self, x, y, atlas):
        self.x = x
        self.y = y
        self.atlas = atlas

class SamplePlus:
    def __init__(self, x, y, y_outer=None, w=None, x_super_res=None, y_super_res=None, y_outer_super_res=None, shape=None):
        self.x = x
        self.y = y
        self.y_outer = y_outer
        self.x_super_res = x_super_res
        self.y_super_res = y_super_res  
        self.w = w
        self.shape = shape


class DatasetAndSupport(object):

    def quick_load_data(self, patch_shape): raise NotImplementedError

    def load_data(self, patch_shape):raise NotImplementedError

    def evaluate(self, target, pred, cfg):raise NotImplementedError

    def save_results(self, target, pred, cfg): raise NotImplementedError

    def update_checkpoint(self, best_so_far, new_value):raise NotImplementedError

 
 
def get_item(item, mode, config):

    x = item.x.cuda()[None]
    y = item.y.cuda()  
    y_outer = item.y_outer.cuda()   
    shape = item.shape  

    # augmentation done only during training
    if mode == DataModes.TRAINING:  # if training do augmentation
        if torch.rand(1)[0] > 0.5:
            x = x.permute([0, 1, 3, 2])
            y = y.permute([0, 2, 1])
            y_outer = y_outer.permute([0, 2, 1])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])
            y_outer = torch.flip(y_outer, dims=[0])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
            y_outer = torch.flip(y_outer, dims=[1])

        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, dims=[3])
            y = torch.flip(y, dims=[2])
            y_outer = torch.flip(y_outer, dims=[2])

        orientation = torch.tensor([0, -1, 0]).float()
        new_orientation = (torch.rand(3) - 0.5) * 2 * np.pi 
        new_orientation = F.normalize(new_orientation, dim=0)
        q = orientation + new_orientation
        q = F.normalize(q, dim=0)
        theta_rotate = stns.stn_quaternion_rotations(q)

        shift = torch.tensor([d / (D // 2) for d, D in zip(2 * (torch.rand(3) - 0.5) * config.augmentation_shift_range, y.shape)])
        theta_shift = stns.shift(shift)
        
        f = 0.1
        scale = 1.0 - 2 * f *(torch.rand(1) - 0.5) 
        theta_scale = stns.scale(scale) 

        theta = theta_rotate @ theta_shift @ theta_scale
 
        x, y, y_outer = stns.transform(theta, x, y, y_outer) 
  

    surface_points_normalized_all = []
    vertices_mc_all = []
    faces_mc_all = [] 
    for i in range(1, config.num_classes):   
        shape = torch.tensor(y.shape)[None].float()
        if mode != DataModes.TRAINING:
            gap = 1
            y_ = clean_border_pixels((y==i).long(), gap=gap)
            vertices_mc, faces_mc = voxel2mesh(y_, gap, shape)
            vertices_mc_all += [vertices_mc]
            faces_mc_all += [faces_mc]
       
     
        y_outer = sample_outer_surface_in_voxel((y==i).long()) 
        surface_points = torch.nonzero(y_outer)
        surface_points = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
        surface_points_normalized = normalize_vertices(surface_points, shape) 
        # surface_points_normalized = y_outer 
      
     
        perm = torch.randperm(len(surface_points_normalized))
        point_count = 3000
        surface_points_normalized_all += [surface_points_normalized[perm[:np.min([len(perm), point_count])]].cuda()]  # randomly pick 3000 points
    
    if mode == DataModes.TRAINING:
        return {   'x': x,  
                   'y_voxels': y, 
                   'surface_points': surface_points_normalized_all, 
                   'unpool':[0, 1, 0, 1, 0]
                }
    else:
        return {   'x': x, 
                   'y_voxels': y, 
                   'vertices_mc': vertices_mc_all,
                   'faces_mc': faces_mc_all,
                   'surface_points': surface_points_normalized_all, 
                   'unpool':[0, 1, 1, 1, 1]}

def sample_outer_surface_in_voxel(volume): 
    # inner surface
    # a = F.max_pool3d(-volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))[0]
    # b = F.max_pool3d(-volume[None,None].float(), kernel_size=(1,3, 1), stride=1, padding=(0, 1, 0))[0]
    # c = F.max_pool3d(-volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))[0] 
    # border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0) 
    # surface = border + volume.float() 

    # outer surface
    a = F.max_pool3d(volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))[0]
    b = F.max_pool3d(volume[None,None].float(), kernel_size=(1,3, 1), stride=1, padding=(0, 1, 0))[0]
    c = F.max_pool3d(volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))[0] 
    border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0) 
    surface = border - volume.float()
    return surface.long()
 

def normalize_vertices(vertices, shape):
    assert len(vertices.shape) == 2 and len(shape.shape) == 2, "Inputs must be 2 dim"
    assert shape.shape[0] == 1, "first dim of shape should be length 1"

    return 2*(vertices/(torch.max(shape)-1) - 0.5)

def sample_to_sample_plus(samples, cfg, datamode):

    new_samples = []
    # surface_point_count = 100
    for sample in samples: 
         
        x = sample.x
        y = sample.y 

        y = (y>0).long()

        center = tuple([d // 2 for d in x.shape]) 
        x = crop(x, cfg.patch_shape, center) 
        y = crop(y, cfg.patch_shape, center)   

        shape = torch.tensor(y.shape)[None].float()
        y_outer = sample_outer_surface_in_voxel(y) 

        new_samples += [SamplePlus(x.cpu(), y.cpu(), y_outer.cpu(), shape=shape)]

    return new_samples

def voxel2mesh(volume, gap, shape):
    '''
    :param volume:
    :param gap:
    :param shape:
    :return:
    '''
    vertices_mc, faces_mc, _, _ = measure.marching_cubes_lewiner(volume.cpu().data.numpy(), 0, step_size=gap, allow_degenerate=False)
    vertices_mc = torch.flip(torch.from_numpy(vertices_mc), dims=[1]).float()  # convert z,y,x -> x, y, z
    vertices_mc = normalize_vertices(vertices_mc, shape)
    faces_mc = torch.from_numpy(faces_mc).long()
    return vertices_mc, faces_mc

def clean_border_pixels(image, gap):
    '''
    :param image:
    :param gap:
    :return:
    '''
    assert len(image.shape) == 3, "input should be 3 dim"

    D, H, W = image.shape
    y_ = image.clone()
    y_[:gap] = 0;
    y_[:, :gap] = 0;
    y_[:, :, :gap] = 0;
    y_[D - gap:] = 0;
    y_[:, H - gap] = 0;
    y_[:, :, W - gap] = 0;

    return y_



 