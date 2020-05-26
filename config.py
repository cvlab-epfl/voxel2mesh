import numpy as np
import torch 

class Config():
    def __init__(self):
        super(Config, self).__init__()


def load_config(exp_id):
      
    cfg = Config()
    ''' Experiment '''
    cfg.experiment_idx = exp_id 
    cfg.trial_id = None
 

    ''' Save at '''
    cfg.save_path = '/cvlabdata2/cvlab/datasets_udaranga/experiments/miccai2020/'
    # cfg.save_path = '/home/wickrama/experiments/shrinknet'

    cfg.save_dir_prefix = 'Experiment_'
 
    cfg.name = 'voxel2mesh'
   

    ''' Dataset ''' 
    cfg.training_set_size = 10  
    cfg.patch_shape = (64, 64, 64)
    cfg.ndims = 3
    cfg.augmentation_shift_range = 10

    ''' Model '''
    cfg.first_layer_channels = 16
    cfg.num_input_channels = 1
    cfg.steps = 4
    cfg.batch_size = 1
    cfg.num_classes = 2
    cfg.batch_norm = True  
    cfg.graph_conv_layer_count = 4

  
    ''' Optimizer '''
    cfg.learning_rate = 1e-4

    ''' Training '''
    cfg.numb_of_itrs = 300000
    cfg.eval_every = 1000
    
    return cfg