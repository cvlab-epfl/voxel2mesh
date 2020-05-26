
import os
import logging

import numpy as np
#import cv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from functools import reduce
import cv2
import sys

volume_suffix = '' 

class DataModes:
    TRAINING = 'training'
    VALIDATION = 'validation'
    TESTING = 'testing'
    ALL = 'all'
    def __init__(self): 
        dataset_splits = [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]


def write_lines(path, lines):
    f = open(path, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
 
def append_line(path, line):
    f = open(path, 'a')
    f.write(line + '\n')
    f.close() 

def pytorch_count_params(model):
  "count number trainable parameters in a pytorch model"
  total_params = sum(reduce(lambda a, b: a*b, x.size()) for x in model.parameters())
  return total_params

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def blend(img, mask):

    img = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_GRAY2RGB)

    rows, cols, d = img.shape
    pre_synaptic = np.zeros((rows, cols, 1))
    pre_synaptic[mask == 1] = 1

    synpase = np.zeros((rows, cols, 1))
    synpase[mask == 2] = 1

    post_synaptic = np.zeros((rows, cols, 1))
    post_synaptic[mask == 3] = 1

    color_mask = np.dstack((synpase, pre_synaptic , post_synaptic))
    color_mask = np.uint8(color_mask*255)

    blended = cv2.addWeighted(img, 0.8, color_mask, 0.2, 0)
    return blended
 

def crop_slices(shape1, shape2):
    slices = [slice((sh1 - sh2) // 2, (sh1 - sh2) // 2 + sh2) for sh1, sh2 in zip(shape1, shape2)]
    return slices

def crop_and_merge(tensor1, tensor2):

    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    slices = tuple(slices)

    return torch.cat((tensor1[slices], tensor2), 1)


def _box_in_bounds(box, image_shape):
    newbox = []
    pad_width = []

    for box_i, shape_i in zip(box, image_shape):
        pad_width_i = (max(0, -box_i[0]), max(0, box_i[1] - shape_i))
        newbox_i = (max(0, box_i[0]), min(shape_i, box_i[1]))

        newbox.append(newbox_i)
        pad_width.append(pad_width_i)

    needs_padding = any(i != (0, 0) for i in pad_width)

    return newbox, pad_width, needs_padding

def crop_indices(image_shape, patch_shape, center):
    box = [(i - ps // 2, i - ps // 2 + ps) for i, ps in zip(center, patch_shape)]
    box, pad_width, needs_padding = _box_in_bounds(box, image_shape)
    slices = tuple(slice(i[0], i[1]) for i in box)
    return slices, pad_width, needs_padding

def crop(image, patch_shape, center, mode='constant'):
    slices, pad_width, needs_padding = crop_indices(image.shape, patch_shape, center)
    patch = image[slices]

    if needs_padding and mode is not 'nopadding':
        if isinstance(image, np.ndarray):
            if len(pad_width) < patch.ndim:
                pad_width.append((0, 0))
            patch = np.pad(patch, pad_width, mode=mode)
        elif isinstance(image, torch.Tensor):
            assert len(pad_width) == patch.dim(), "not supported"
            # [int(element) for element in np.flip(np.array(pad_width).flatten())]
            patch = F.pad(patch, tuple([int(element) for element in np.flip(np.array(pad_width), axis=0).flatten()]), mode=mode)

    return patch

 


def blend(img, labels, num_classes):
    colors = torch.tensor([[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255]]).cuda().float()


    img = img[..., None].repeat(1, 1, 1, 3)
    masks = torch.zeros_like(img)
    for cls in range(1, num_classes):
        masks += torch.ones_like(img) * colors[cls] * (labels == cls).float()[:, :, :, None]

    overlay = np.uint8((255 * img * 0.8 + masks * 0.2).data.cpu().numpy())
    return overlay

def blend_cpu(img, labels, num_classes):
    colors = torch.tensor([[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255]]).float()


    img = img[..., None].repeat(1, 1, 1, 3)
    masks = torch.zeros_like(img)
    for cls in range(1, num_classes):
        masks += torch.ones_like(img) * colors[cls] * (labels == cls).float()[:, :, :, None]

    overlay = np.uint8((255 * img * 0.8 + masks * 0.2).data.numpy())
    return overlay






