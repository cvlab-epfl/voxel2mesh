 
import os
GPU_index = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_index

  
import logging
import torch
import numpy as np
from train import Trainer
from evaluate import Evaluator  
from data.chaos import Chaos
from data.hippocampus import Hippocampus

from shutil import copytree, ignore_patterns
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.utils_common import DataModes
import wandb
from IPython import embed 
from utils.utils_common import mkdir

from config import load_config
from model.voxel2mesh import Voxel2Mesh as network

 
def main():
 
     

    # Initialize
    cfg = load_config(None)
 
  
    print("Pre-process data") 
    data_obj = cfg.data_obj  

    # Run pre-processing
    data = data_obj.pre_process_dataset(cfg)
  

if __name__ == "__main__": 
    main()