import numpy as np
import torch
  
import time
from IPython import embed
from scipy.io import savemat 

def read_obj(filepath):
    vertices = []
    faces = [] 
    normals = []   
    with open(filepath) as fp:
        line = fp.readline() 
        cnt = 1 
        while line: 
            if line[0] is not '#': 
                cnt = cnt + 1 
                values = [float(x) for x in line.split('\n')[0].split(' ')[1:]] 
                if line[:2] == 'vn':  
                    normals.append(values)
                elif line[0] == 'v':
                    vertices.append(values)
                elif line[0] == 'f':
                    faces.append(values) 
            line = fp.readline()
        vertices = np.array(vertices)
        normals = np.array(normals)
        faces = np.array(faces)
        faces = np.int64(faces) - 1
        if len(normals) > 0:
            return vertices, faces, normals
        else:
            return vertices, faces


def save_to_obj(filepath, points, faces, normals=None): 
    with open(filepath, 'w') as file:
        vals = ''
        for i, point in enumerate(points[0]):
            point = point.data.cpu().numpy()
            vals += 'v ' + ' '.join([str(val) for val in point]) + '\n'
        if normals is not None:
            for i, normal in enumerate(normals[0]):
                normal = normal.data.cpu().numpy()
                vals += 'vn ' + ' '.join([str(val) for val in normal]) + '\n'
        if len(faces) > 0:
            for i, face in enumerate(faces[0]):
                face = face.data.cpu().numpy()
                vals += 'f ' + ' '.join([str(val+1) for val in face]) + '\n'
        file.write(vals)