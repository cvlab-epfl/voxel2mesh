from ctypes import *
import ctypes
import numpy as np

''' TODO: to many functions, combine them'''
def cuda_get_rasterize():
    dll = ctypes.CDLL('/cvlabdata2/home/wickrama/projects/U-Net/Experiments/meshnet/mnet/kernel.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_rasterize
    func.argtypes = [POINTER(c_int), POINTER(c_float), POINTER(c_int), POINTER(c_float), c_size_t, c_size_t, c_size_t, c_size_t, c_size_t]

    ctypes._reset_cache()
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_rasterize = cuda_get_rasterize()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_rasterize(grid, vertices, faces, D, H, W, N_vertices, N_faces, debug):
    grid_p = grid.ctypes.data_as(POINTER(c_int))
    vertices_p = vertices.ctypes.data_as(POINTER(c_float))
    faces_p = faces.ctypes.data_as(POINTER(c_int))
    debug_p = debug.ctypes.data_as(POINTER(c_float))

    __cuda_rasterize(grid_p, vertices_p, faces_p, debug_p, D, H, W, N_vertices, N_faces)
 


# testing, sum of two arrays of ones and output head part of resulting array
def rasterize_gpu(vertices, faces, grid_size):

    D, H, W = grid_size
    N_vertices = len(vertices)
    N_faces = len(faces)
    volume = np.zeros(grid_size).astype('int32')
    debug = np.zeros(grid_size).astype('float32')
    vertices = vertices.astype('float32')
    faces = faces.astype('int32')

    cuda_rasterize(volume, vertices, faces, D, H, W, N_vertices, N_faces, debug)


    return volume, debug
 

def run_rasterize(vertices, faces_, grid_size):
    v = [vertices[faces_[:, i], :] for i in range(3)]
    face_areas = np.abs(np.cross(v[2] - v[0], v[1] - v[0]) / 2)
    face_areas = np.linalg.norm(face_areas, axis=1)
    faces = faces_[face_areas > 0]

    labels, _ = rasterize_gpu(vertices, faces, grid_size=grid_size)

    return labels


# 
# D, H, W output volume dimension
# (input)
# vertices: N x 3
# faces: F x 3

# (output)
# y_voxels: D x H x W dim image volume  
D = H = W = 64

# torch.flip -> vertices are (x,y,z) coordinates. In tensors axis order is reversed (z, y, x) [that's that's the standard]
# ([D-1, H-1, W-1])[None].float() * (v + 1)/2 -> in my case vertices are in the range (-1,1). I shift them to tensor index range (0 to D/H/W - 1). 
# if your vertices are already in that range, you don't need this
v = torch.tensor([D-1, H-1, W-1])[None].float() * (torch.flip(vertices.clone(), [1]) + 1)/2 

# remember to round, otherwise rasterize does casting to int and introduce a single pixel error to some vertices
v = torch.round(v)

# transfeer to cpu 
v = v.data.cpu().numpy()

f = faces.clone().numpy()

# run rasterize twice -> dirty trick to free gpu memomry. this is required when
# you run rasterize for multiple meshes. It overwrites on the volume from last mesh when computing 
# the new volume.  To clear it, we first rasterize with a zero-mesh (v*0).
# I think It's a issue with gc when and python-c++-cuda. You are free to fix it if u like
y_voxels = run_rasterize(v * 0, f, grid_size=(D, H, W))
y_voxels = run_rasterize(v, f, grid_size=(D, H, W))