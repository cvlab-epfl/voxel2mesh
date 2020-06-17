from torch.utils.cpp_extension import load
rasterize_cuda = load(
    'rasterize_cuda', ['./utils/rasterize/rasterize_cuda.cpp', './utils/rasterize/rasterize_cuda_kernel.cu'], verbose=True)
# rasterize_cuda = load(
#     'rasterize_cuda', ['./utils/rasterize/rasterize_cuda.cpp', './utils/rasterize/dst_tf_cuda_kernel.cu'], verbose=True)


import math
from torch import nn
from torch.autograd import Function
import torch

import rasterize_cuda
from IPython import embed


torch.manual_seed(42)


class RasterizeFunction(Function):
    @staticmethod
    def forward(ctx, vertices, faces, shape):
        # embed()

        N, _, _ = vertices.shape
        D, H, W = shape 
        shape = torch.tensor([D,H,W]).int().cuda()
        volume = []

        if not vertices.is_cuda:
            vertices = vertices.cuda()

        if not faces.is_cuda:
            faces = faces.cuda()    
                 
        for vertices_, faces_ in zip(vertices, faces):
            v = (shape[None].float() - 1) * (vertices_.clone() + 1)/2 
            # v = vertices_.clone()
            v = torch.round(v).float()
            f = faces_.int() 
 
            volume_ = rasterize_cuda.forward(v, f, shape)[0].float()[None]  
            volume += [volume_]
        
        volume = torch.cat(volume, dim=0)
        volume.requires_grad = True
        # ctx.save_for_backward(*variables) 
        ctx.vertices = vertices
        ctx.faces = faces
        ctx.shape = shape
        ctx.volume = volume

        return volume

    @staticmethod
    def backward(ctx, grad_output):
        vertices = ctx.vertices
        faces = ctx.faces
        shape = ctx.shape
        volume = ctx.volume

        grad_volume = grad_output.contiguous()
        D, H, W = shape 
        shape = torch.tensor([D,H,W]).int().cuda()
        grad_vertices = []
        # embed()
                 
        for output_, grad_volume_, vertices_, faces_ in zip(volume, grad_volume, vertices, faces):
            v = (shape[None].float() - 1) * (vertices_.clone() + 1)/2 
            # v = vertices_.clone()
            v = torch.round(v).float()
            f = faces_.int() 
 
            grad_vertices_ = rasterize_cuda.backward(output_, grad_volume_, v, f, shape)[0].float()[None]  
            grad_vertices += [grad_vertices_]
        
        grad_vertices = torch.cat(grad_vertices, dim=0)
        # grad_vertices = vertices
        grad_faces = grad_shape = None
        return grad_vertices, grad_faces, grad_shape


class Rasterize(nn.Module):
    def __init__(self, shape):
        super(Rasterize, self).__init__() 
        self.shape = shape
 

    def forward(self, vertices, faces): 
        return RasterizeFunction.apply(vertices, faces, self.shape)


class RasterizeCPU(nn.Module):
    def __init__(self, shape):
        super(Rasterize, self).__init__() 
        self.shape = shape
 

    def forward(self, vertices, faces): 
        return RasterizeFunction.apply(vertices, faces, self.shape)
# embed()
