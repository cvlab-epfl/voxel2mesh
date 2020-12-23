import numpy as np
from skimage import io
from data.data import DatasetAndSupport, get_item, sample_to_sample_plus

# from evaluate.standard_metrics import jaccard_index, chamfer_weighted_symmetric, chamfer_directed
from utils.metrics import jaccard_index, chamfer_weighted_symmetric, chamfer_directed
from utils.utils_common import crop, DataModes, crop_indices, blend
# from utils.utils_common import invfreq_lossweights, crop, DataModes, crop_indices, blend, volume_suffix
# from utils.utils_mesh import sample_outer_surface, get_extremity_landmarks, voxel2mesh, clean_border_pixels

from utils import stns
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
import h5py
import sys

class Sample:
    def __init__(self, x, y, atlas):
        self.x = x
        self.y = y
        self.atlas = atlas

class HippocampusDataset(Dataset):

    def __init__(self, data, cfg, mode): 
        self.data = data 

        self.cfg = cfg
        self.mode = mode
 

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx] 
        return get_item(item, self.mode, self.cfg)



class Hippocampus(DatasetAndSupport):


    def quick_load_data(self, cfg, trial_id): 

        data_root = cfg.dataset_path 
        data = {}  
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):
            with open(data_root + '/pre_computed_data_' + datamode + '.pickle', 'rb') as handle:
                samples = pickle.load(handle) 
                new_samples = sample_to_sample_plus(samples, cfg, datamode)
                data[datamode] = HippocampusDataset(new_samples, cfg, datamode) 
                 

        return data

    def pre_process_dataset(self, cfg):
        '''
         :
        ''' 
        print('Data pre-processing - Hippocampus Dataset')

        data_root = cfg.dataset_path
        down_sample_shape = cfg.patch_shape  # (224, 224, 224)
        largest_image_size = (64, 64, 64)
        # data = self.load_nii(data_root, cfg, HippocampusDataset, down_sample_shape, largest_image_size)

        samples = [dir for dir in os.listdir('{}/imagesTr'.format(data_root))]

        print('Load data...')

        inputs = []
        labels = []

        vals = []
        sizes = [] 
        for itr, sample in enumerate(samples):

            if '.npy' in sample and '._' not in sample:

                x, y = self.read_sample(data_root, sample, down_sample_shape, largest_image_size)
                inputs += [x.cpu()]
                labels += [y.cpu()]
 
        
        inputs_ = [i[None].data.numpy() for i in inputs] 
        labels_ = [i[None].data.numpy() for i in labels]
        inputs_ = np.concatenate(inputs_, axis=0)
        labels_ = np.concatenate(labels_, axis=0)

        hf = h5py.File(data_root + '/data.h5', 'w') 
        hf.create_dataset('inputs', data=inputs_)
        hf.create_dataset('labels', data=labels_)
        hf.close()
 
        print('Saving data...')

        np.random.seed(0)
        perm = np.random.permutation(len(inputs)) 
        counts = [perm[:len(inputs)//2], perm[len(inputs)//2:]]

        data = {}
        # down_sample_shape = (32, 128, 128)
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):

            samples = []

            for j in counts[i]:
                x = inputs[j]
                y = labels[j]

                samples.append(Sample(x, y, None))

            with open(data_root + '/pre_computed_data_' + datamode + '.pickle', 'wb') as handle:
                pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # data[datamode] = HippocampusDataset(samples, cfg, datamode)

        print('\nPre-processing complete') 
        return data

    def evaluate(self, target, pred, cfg):
        results = {}


        if target.voxel is not None:
            val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
            results['jaccard'] = val_jaccard

        if target.points is not None:
            target_points = target.points
            pred_points = pred.mesh
            val_chamfer_weighted_symmetric = np.zeros(len(target_points))

            for i in range(len(target_points)):
                val_chamfer_weighted_symmetric[i] = chamfer_weighted_symmetric(target_points[i].cpu(), pred_points[i]['vertices'])

            results['chamfer_weighted_symmetric'] = val_chamfer_weighted_symmetric

        return results

    def update_checkpoint(self, best_so_far, new_value):

        if 'chamfer_weighted_symmetric' in new_value[DataModes.TESTING]:
            key = 'chamfer_weighted_symmetric'
            new_value = new_value[DataModes.TESTING][key]

            if best_so_far is None:
                return True
            else:
                best_so_far = best_so_far[DataModes.TESTING][key]
                return True if np.mean(new_value) < np.mean(best_so_far) else False

        elif 'jaccard' in new_value[DataModes.TESTING]:
            key = 'jaccard'
            new_value = new_value[DataModes.TESTING][key]

            if best_so_far is None:
                return True
            else:
                best_so_far = best_so_far[DataModes.TESTING][key]
                return True if np.mean(new_value) > np.mean(best_so_far) else False

 
    def read_sample(self, data_root, sample, out_shape, pad_shape):
        x = np.load('{}/imagesTr/{}'.format(data_root, sample))
        y = np.load('{}/labelsTr/{}'.format(data_root, sample))

        D, H, W = x.shape
        center_z, center_y, center_x = D // 2, H // 2, W // 2
        D, H, W = pad_shape
        x = crop(x, (D, H, W), (center_z, center_y, center_x))
        y = crop(y, (D, H, W), (center_z, center_y, center_x))

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        x = F.interpolate(x[None, None], out_shape, mode='trilinear', align_corners=False)[0, 0]
        y = F.interpolate(y[None, None].float(), out_shape, mode='nearest')[0, 0].long()

        return x, y
