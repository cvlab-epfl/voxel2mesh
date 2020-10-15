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
 

    ''' 
    **** Paths *****
    save_path: results will be saved at this location
    dataset_path: dataset must be stored here.

    Note: During the first run use load_data function. It will do the necessary preprocessing and save the files at the same location.
    After that, you can use quick_load_data function to load them. This function is called in main.py

    '''

    # cfg.save_path = '/your/path/to/experiments/miccai2020/' # results will be saved here
    # cfg.dataset_path = '/your/path/to/dataset'


    # example
    cfg.dataset_path = '/cvlabsrc1/cvlab/datasets_udaranga/datasets/3d/chaos/Train_Sets/CT'
    cfg.save_path = '/cvlabdata2/cvlab/datasets_udaranga/experiments/vmnet/'
    cfg.save_dir_prefix = 'Experiment_' # prefix for experiment folder
 
    cfg.name = 'voxel2mesh'
   

    ''' Dataset ''' 
    cfg.training_set_size = 10  

    # input should be cubic. Otherwise, input should be padded accordingly.
    cfg.patch_shape = (64, 64, 64) 
    

    cfg.ndims = 3
    cfg.augmentation_shift_range = 10

    ''' Model '''
    cfg.first_layer_channels = 16
    cfg.num_input_channels = 1
    cfg.steps = 4

    # Only supports batch size 1 at the moment. 
    cfg.batch_size = 1 


    cfg.num_classes = 2
    cfg.batch_norm = True  
    cfg.graph_conv_layer_count = 4

  
    ''' Optimizer '''
    cfg.learning_rate = 1e-4

    ''' Training '''
    cfg.numb_of_itrs = 300000
    cfg.eval_every = 1000 # saves results to disk

    # ''' Rreporting '''
    # cfg.wab = True # use weight and biases for reporting
    
    return cfg