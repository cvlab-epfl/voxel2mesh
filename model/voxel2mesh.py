import torch.nn as nn
import torch 
import torch.nn.functional as F 

from pytorch3d.structures import Meshes 
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance,  mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)

import numpy as np
from itertools import product, combinations, chain
from scipy.spatial import ConvexHull

from IPython import embed 
import time 

from utils.utils_common import crop_and_merge  
from utils.utils_voxel2mesh.graph_conv import adjacency_matrix, Features2Features, Feature2VertexLayer 
from utils.utils_voxel2mesh.feature_sampling import LearntNeighbourhoodSampling 
from utils.utils_voxel2mesh.file_handle import read_obj 


from utils.utils_voxel2mesh.unpooling import uniform_unpool, adoptive_unpool

from utils.utils_unet import UNetLayer


  
 
 
class Voxel2Mesh(nn.Module):
    """ Voxel2Mesh  """
 
    def __init__(self, config):
        super(Voxel2Mesh, self).__init__()

        self.config = config
          
        self.max_pool = nn.MaxPool3d(2) if config.ndims == 3 else nn.MaxPool2d(2) 

        ConvLayer = nn.Conv3d if config.ndims == 3 else nn.Conv2d
        ConvTransposeLayer = nn.ConvTranspose3d if config.ndims == 3 else nn.ConvTranspose2d
        batch_size = config.batch_size
 

        '''  Down layers '''
        down_layers = [UNetLayer(config.num_input_channels, config.first_layer_channels, config.ndims)]
        for i in range(1, config.steps + 1):
            graph_conv_layer = UNetLayer(config.first_layer_channels * 2 ** (i - 1), config.first_layer_channels * 2 ** i, config.ndims)
            down_layers.append(graph_conv_layer)
        self.down_layers = down_layers
        self.encoder = nn.Sequential(*down_layers)
 

        ''' Up layers ''' 
        self.skip_count = []
        self.latent_features_coount = []
        for i in range(config.steps+1):
            self.skip_count += [config.first_layer_channels * 2 ** (config.steps-i)] 
            self.latent_features_coount += [32]

        dim = 3

        up_std_conv_layers = []
        up_f2f_layers = []
        up_f2v_layers = []
        for i in range(config.steps+1):
            graph_unet_layers = []
            feature2vertex_layers = []
            skip = LearntNeighbourhoodSampling(config, self.skip_count[i], i)
            # lyr = Feature2VertexLayer(self.skip_count[i])
            if i == 0:
                grid_upconv_layer = None
                grid_unet_layer = None
                for k in range(config.num_classes-1):
                    graph_unet_layers += [Features2Features(self.skip_count[i] + dim, self.latent_features_coount[i], hidden_layer_count=config.graph_conv_layer_count)] # , graph_conv=GraphConv

            else:
                grid_upconv_layer = ConvTransposeLayer(in_channels=config.first_layer_channels   * 2**(config.steps - i+1), out_channels=config.first_layer_channels * 2**(config.steps-i), kernel_size=2, stride=2)
                grid_unet_layer = UNetLayer(config.first_layer_channels * 2**(config.steps - i + 1), config.first_layer_channels * 2**(config.steps-i), config.ndims, config.batch_norm)
                for k in range(config.num_classes-1):
                    graph_unet_layers += [Features2Features(self.skip_count[i] + self.latent_features_coount[i-1] + dim, self.latent_features_coount[i], hidden_layer_count=config.graph_conv_layer_count)] #, graph_conv=GraphConv if i < config.steps else GraphConvNoNeighbours

            for k in range(config.num_classes-1):
                feature2vertex_layers += [Feature2VertexLayer(self.latent_features_coount[i], 3)] 
 

            up_std_conv_layers.append((skip, grid_upconv_layer, grid_unet_layer))
            up_f2f_layers.append(graph_unet_layers)
            up_f2v_layers.append(feature2vertex_layers)
        
 

        self.up_std_conv_layers = up_std_conv_layers
        self.up_f2f_layers = up_f2f_layers
        self.up_f2v_layers = up_f2v_layers

        self.decoder_std_conv = nn.Sequential(*chain(*up_std_conv_layers))
        self.decoder_f2f = nn.Sequential(*chain(*up_f2f_layers))
        self.decoder_f2v = nn.Sequential(*chain(*up_f2v_layers))

        ''' Final layer (for voxel decoder)'''
        self.final_layer = ConvLayer(in_channels=config.first_layer_channels, out_channels=config.num_classes, kernel_size=1)

        sphere_path='./spheres/icosahedron_{}.obj'.format(162)
        sphere_vertices, sphere_faces = read_obj(sphere_path)
        sphere_vertices = torch.from_numpy(sphere_vertices).cuda().float()
        self.sphere_vertices = sphere_vertices/torch.sqrt(torch.sum(sphere_vertices**2,dim=1)[:,None])[None]
        self.sphere_faces = torch.from_numpy(sphere_faces).cuda().long()[None]


 
  
    def forward(self, data):
         
        x = data['x'] 
        unpool_indices = data['unpool'] 

        sphere_vertices = self.sphere_vertices.clone()
        vertices = sphere_vertices.clone()
        faces = self.sphere_faces.clone() 
        batch_size = self.config.batch_size  
 
        # first layer
        x = self.down_layers[0](x)
        down_outputs = [x]

        # down layers
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x) 
            down_outputs.append(x)

  
        A, D = adjacency_matrix(vertices, faces)
        pred = [None] * self.config.num_classes 
        for k in range(self.config.num_classes-1):
            pred[k] = [[vertices.clone(), faces.clone(), None, None, sphere_vertices.clone()]]

 
        for i, ((skip_connection, grid_upconv_layer, grid_unet_layer), up_f2f_layers, up_f2v_layers, down_output, skip_amount, do_unpool) in enumerate(zip(self.up_std_conv_layers, self.up_f2f_layers, self.up_f2v_layers, down_outputs[::-1], self.skip_count, unpool_indices)):
            if grid_upconv_layer is not None and i > 0:
                x = grid_upconv_layer(x)
                x = crop_and_merge(down_output, x)
                x = grid_unet_layer(x)
            elif grid_upconv_layer is None:
                x = down_output
          

            for k in range(self.config.num_classes-1):

            	# load mesh information from previous iteratioin for class k
                vertices = pred[k][i][0]
                faces = pred[k][i][1]
                latent_features = pred[k][i][2]
                sphere_vertices = pred[k][i][4]
                graph_unet_layer = up_f2f_layers[k]
                feature2vertex = up_f2v_layers[k]
 
                if do_unpool[0] == 1:
                    faces_prev = faces
                    _, N_prev, _ = vertices.shape 

                    # Get candidate vertices using uniform unpool
                    vertices, faces_ = uniform_unpool(vertices, faces)  
                    latent_features, _ = uniform_unpool(latent_features, faces)  
                    sphere_vertices, _ = uniform_unpool(sphere_vertices, faces) 
                    faces = faces_  

                
                A, D = adjacency_matrix(vertices, faces)
                skipped_features = skip_connection(x[:, :skip_amount], vertices)      
                      
                latent_features = torch.cat([latent_features, skipped_features, vertices], dim=2) if latent_features is not None else torch.cat([skipped_features, vertices], dim=2)
 
                latent_features = graph_unet_layer(latent_features, A, D, vertices, faces)
                deltaV = feature2vertex(latent_features, A, D, vertices, faces)
                vertices = vertices + deltaV 
                
                if do_unpool[0] == 1:
                    # Discard the vertices that were introduced from the uniform unpool and didn't deform much
                    vertices, faces, latent_features, sphere_vertices = adoptive_unpool(vertices, faces_prev, sphere_vertices, latent_features, N_prev)

                voxel_pred = self.final_layer(x) if i == len(self.up_std_conv_layers)-1 else None

                pred[k] += [[vertices, faces, latent_features, voxel_pred, sphere_vertices]]
 
        return pred


    def loss(self, data, epoch):

         
        pred = self.forward(data)  
        # embed()
        

         
        CE_Loss = nn.CrossEntropyLoss() 
        ce_loss = CE_Loss(pred[0][-1][3], data['y_voxels'])


        chamfer_loss = torch.tensor(0).float().cuda()
        edge_loss = torch.tensor(0).float().cuda()
        laplacian_loss = torch.tensor(0).float().cuda()
        normal_consistency_loss = torch.tensor(0).float().cuda()  

        for c in range(self.config.num_classes-1):
            target = data['surface_points'][c].cuda() 
            for k, (vertices, faces, _, _, _) in enumerate(pred[c][1:]):
      
                pred_mesh = Meshes(verts=list(vertices), faces=list(faces))
                pred_points = sample_points_from_meshes(pred_mesh, 3000)
                
                chamfer_loss +=  chamfer_distance(pred_points, target)[0]
                laplacian_loss +=   mesh_laplacian_smoothing(pred_mesh, method="uniform")
                normal_consistency_loss += mesh_normal_consistency(pred_mesh) 
                edge_loss += mesh_edge_loss(pred_mesh) 

        
        
 
        loss = 1 * chamfer_loss + 1 * ce_loss + 0.1 * laplacian_loss + 1 * edge_loss + 0.1 * normal_consistency_loss
 
        log = {"loss": loss.detach(),
               "chamfer_loss": chamfer_loss.detach(), 
               "ce_loss": ce_loss.detach(),
               "normal_consistency_loss": normal_consistency_loss.detach(),
               "edge_loss": edge_loss.detach(),
               "laplacian_loss": laplacian_loss.detach()}
        return loss, log


 

 

