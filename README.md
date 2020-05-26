# Voxel2Mesh: 3D Mesh Model Generation from Volumetric Data

This repository is a PyTorch implementation of Voxel2Mesh architecture proposed in *Voxel2Mesh: 3D Mesh Model Generation from Volumetric Data*, Udaranga Wickramasinghe, Edoardo Remelli, Graham Knott, Pascal Fua, MICCAI 2020.

## Abstract

CNN-based volumetric methods that label individual voxels now dominate the field of biomedical segmentation. However, 3D surface representations are often required for proper analysis. They can be obtained by post-processing the labeled volumes which typically introduces artifacts and prevents end-to-end training. In this paper, we therefore introduce a novel architecture that goes directly from 3D image volumes to 3D surfaces without post-processing and with better accuracy than current methods. We evaluate it on Electron Microscopy and MRI brain images as well as CT liver scans. We will show that it outperforms state-of-the-art segmentation methods.
