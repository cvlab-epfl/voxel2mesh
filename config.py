import numpy as np
import torch 

from data.chaos import Chaos
from data.hippocampus import Hippocampus

class Config():
    def __init__(self):
        super(Config, self).__init__()


def load_config(exp_id):
      
    cfg = Config()
    ''' Experiment '''
    cfg.experiment_idx = exp_id 
    cfg.trial_id = None
    
    cfg.save_dir_prefix = 'Experiment_' # prefix for experiment folder
    cfg.name = 'voxel2mesh'

    ''' 
    **************************************** Paths ****************************************
    save_path: results will be saved at this location
    dataset_path: dataset must be stored here.
    '''
    cfg.save_path = None    # UPDATE HERE <<<<<<<<<<<<<<<<<<<<<<
    cfg.dataset_path = None # UPDATE HERE <<<<<<<<<<<<<<<<<<<<<<
    
    # cfg.save_path = '/your/path/to/experiments/miccai2020/' # results will be saved here
    # cfg.dataset_path = '/your/path/to/dataset' # path to the dataset

    # Initialize data object for. 
    # Hippocampus() for hippocampus and Chaos() for liver dataset. 

    cfg.data_obj = None     # UPDATE HERE <<<<<<<<<<<<<<<<<<<<<<
    # cfg.data_obj = Chaos() 
    # cfg.data_obj = Hippocampus()


    assert cfg.save_path != None, "Set cfg.save_path in config.py"
    assert cfg.dataset_path != None, "Set cfg.dataset_path in config.py"
    assert cfg.data_obj != None, "Set cfg.data_obj in config.py"

    ''' 
    ************************************************************************************************
    ''' 




    ''' Dataset '''  
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