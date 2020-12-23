import numpy as np
from skimage import io
from data.data import get_item, sample_outer_surface_in_voxel, sample_to_sample_plus

import sys
from utils.metrics import jaccard_index, chamfer_weighted_symmetric, chamfer_directed
from utils.utils_common import crop, DataModes, crop_indices, blend
# from utils.utils_mesh import sample_outer_surface, get_extremity_landmarks, voxel2mesh, clean_border_pixels, sample_outer_surface_in_voxel, normalize_vertices 

# from utils import stns
from torch.utils.data import Dataset
import torch
from sklearn.decomposition import PCA
import pickle
import torch.nn.functional as F
from numpy.linalg import norm
import itertools as itr
import torch
from scipy import ndimage
import os
from IPython import embed
import pydicom

class Sample:
    def __init__(self, x, y, atlas=None):
        self.x = x
        self.y = y
        self.atlas = atlas

class SamplePlus:
    def __init__(self, x, y, y_outer=None, x_super_res=None, y_super_res=None, y_outer_super_res=None, shape=None):
        self.x = x
        self.y = y
        self.y_outer = y_outer
        self.x_super_res = x_super_res
        self.y_super_res = y_super_res  
        self.shape = shape

  
class ChaosDataset():

    def __init__(self, data, cfg, mode): 
        self.data = data  

        self.cfg = cfg
        self.mode = mode
 

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx] 
        return get_item(item, self.mode, self.cfg) 

  

class Chaos():




    def pick_surface_points(self, y_outer, point_count):
        idxs = torch.nonzero(y_outer) 
        perm = torch.randperm(len(idxs))

        y_outer = y_outer * 0  
        idxs = idxs[perm[:point_count]]
        y_outer[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
        return y_outer

    def quick_load_data(self, cfg, trial_id):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        down_sample_shape = cfg.patch_shape

        data_root = cfg.dataset_path
        data = {}
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):
            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'rb') as handle:
                samples = pickle.load(handle)
                new_samples = sample_to_sample_plus(samples, cfg, datamode)
                data[datamode] = ChaosDataset(new_samples, cfg, datamode) 

        return data

    def pre_process_dataset(self, cfg):
        '''
         :
        '''
 
        data_root = cfg.dataset_path
        samples = [dir for dir in os.listdir(data_root)]
 
        pad_shape = (384, 384, 384)
        inputs = []
        labels = []

        print('Data pre-processing - Chaos Dataset')
        for sample in samples:
            if 'pickle' not in sample:
                print('.', end='', flush=True)
                x = [] 
                images_path = [dir for dir in os.listdir('{}/{}/DICOM_anon'.format(data_root, sample)) if 'dcm' in dir]
                for image_path in images_path:
                    file = pydicom.dcmread('{}/{}/DICOM_anon/{}'.format(data_root, sample, image_path))
                    x += [file.pixel_array] 

                d_resolution = file.SliceThickness
                h_resolution, w_resolution = file.PixelSpacing 
                x = np.float32(np.array(x))

 
                D, H, W = x.shape
                D = int(D * d_resolution) #  
                H = int(H * h_resolution) # 
                W = int(W * w_resolution)  #  
                # we resample such that 1 pixel is 1 mm in x,y and z directiions
                base_grid = torch.zeros((1, D, H, W, 3))
                w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
                h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
                d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)
                base_grid[:, :, :, :, 0] = w_points
                base_grid[:, :, :, :, 1] = h_points
                base_grid[:, :, :, :, 2] = d_points
                
                grid = base_grid.cuda()
                 
                
                x = torch.from_numpy(x).cuda()
                x = F.grid_sample(x[None, None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0, 0]
                x = x.data.cpu().numpy() 
                #----
                 
                x = np.float32(x) 
                mean_x = np.mean(x)
                std_x = np.std(x)

                D, H, W = x.shape
                center_z, center_y, center_x = D // 2, H // 2, W // 2
                D, H, W = pad_shape
                x = crop(x, (D, H, W), (center_z, center_y, center_x))  
  
                # normalize x
                x = (x - mean_x)/std_x
                x = torch.from_numpy(x)
                inputs += [x]
                 
                #----
 
                y = [] 
                images_path = [dir for dir in os.listdir('{}/{}/Ground'.format(data_root, sample)) if 'png' in dir]
                for image_path in images_path:
                    file = io.imread('{}/{}/Ground/{}'.format(data_root, sample, image_path))
                    y += [file]  
                 
                y = np.array(y) 
                y = np.int64(y) 

                y = torch.from_numpy(y).cuda()
                y = F.grid_sample(y[None, None].float(), grid, mode='nearest', padding_mode='border', align_corners=True)[0, 0]
                y = y.data.cpu().numpy()

                 
               
                y = np.int64(y)
                y = crop(y, (D, H, W), (center_z, center_y, center_x))  
                  
                 
                y = torch.from_numpy(y/255) 
                  
                labels += [y]

        print('\nSaving pre-processed data to disk')
        np.random.seed(0)
        perm = np.random.permutation(len(inputs)) 
        counts = [perm[:len(inputs)//2], perm[len(inputs)//2:]]
 
        data = {}
        down_sample_shape = cfg.patch_shape

        input_shape = x.shape
        scale_factor = (np.max(down_sample_shape)/np.max(input_shape))

        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):

            samples = []
 

            for j in counts[i]: 
                print('.',end='', flush=True)
                x = inputs[j]
                y = labels[j]

                x = F.interpolate(x[None, None], scale_factor=scale_factor, mode='trilinear')[0, 0]
                y = F.interpolate(y[None, None].float(), scale_factor=scale_factor, mode='nearest')[0, 0].long()

                samples.append(Sample(x, y)) 

            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'wb') as handle:
                pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            data[datamode] = ChaosDataset(samples, cfg, datamode)
        
        print('Pre-processing complete') 
        return data
 
    def evaluate(self, target, pred, cfg):
        results = {}


        if target.voxel is not None: 
            val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
            results['jaccard'] = val_jaccard

        if target.mesh is not None:
            target_points = target.points
            pred_points = pred.mesh
            val_chamfer_weighted_symmetric = np.zeros(len(target_points))

            for i in range(len(target_points)):
                val_chamfer_weighted_symmetric[i] = chamfer_weighted_symmetric(target_points[i].cpu(), pred_points[i]['vertices'])

            results['chamfer_weighted_symmetric'] = val_chamfer_weighted_symmetric

        return results

    def update_checkpoint(self, best_so_far, new_value):

        key = 'jaccard'
        new_value = new_value[DataModes.TESTING][key]

        if best_so_far is None:
            return True
        else:
            best_so_far = best_so_far[DataModes.TESTING][key]
            return True if np.mean(new_value) > np.mean(best_so_far) else False



 