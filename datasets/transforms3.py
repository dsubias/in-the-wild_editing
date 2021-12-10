import numpy as np
from PIL import Image
import random

import torch
#from torchvision import transforms as T
#from torchvision.transforms import functional as F
from opencv_transforms import functional as F
from opencv_transforms import transforms as T



# def pad_if_smaller(img, size, fill=0):
#     min_size = min(img.size)
#     if min_size < size:
#         ow, oh = img.size
#         padh = size - oh if oh < size else 0
#         padw = size - ow if ow < size else 0
#         img = F.pad(img, (0, 0, padw, padh), fill=fill)
#     return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, normals, illum):
        for t in self.transforms:
            image, normals, illum = t(image, normals, illum)
        return image, normals, illum


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, normals, illum):
        image = F.center_crop(image, self.size)
        normals = F.center_crop(normals, self.size)
        illum = F.center_crop(illum, self.size)
        return image, normals, illum

class Resize(object): 
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = (size,size) #TODO according to what is size
        self.interpolation = interpolation

    def __call__(self, image, normals, illum): #TODO warning f
        image = F.resize(image, self.size, self.interpolation)
        normals = F.resize(normals, self.size, self.interpolation)
        illum = F.resize(illum, self.size, self.interpolation)
        return image, normals, illum

class ToTensor(object):
    def __call__(self, image, normals, illum):
        image = F.to_tensor(image)
        normals = F.to_tensor(normals)
        illum = F.to_tensor(illum)
        return image, normals, illum

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, normals, illum):
        image = F.normalize(image, mean=self.mean, std=self.std)
        normals = F.normalize(normals, mean=self.mean, std=self.std)
        illum = F.normalize(illum, mean=self.mean, std=self.std)
        return image, normals, illum

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, normals, illum):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            normals = F.hflip(normals) #TODO change normals
            illum = F.hflip(illum) 
        return image, normals, illum

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob


    def __call__(self, image, normals, illum):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            normals = F.vflip(normals) 
            illum = F.vflip(illum) 
            #TODO change normals
        return image, normals, illum

class RandomCrop(object):
    def __init__(self, size):
        self.size = size


    def __call__(self, image, normals, illum):
        #image = pad_if_smaller(image, self.size)
        #normals = pad_if_smaller(normals, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        normals = F.crop(normals, *crop_params)
        illum = F.crop(illum, *crop_params)
        return image, normals, illum


class RandomResize(object):
    def __init__(self, low, high, interpolation=Image.BILINEAR):
        self.low = low
        self.high = high
        self.interpolation = interpolation


    def __call__(self, image, normals, illum): 
        size = np.random.randint(self.low, self.high)
        image = F.resize(image, (size,size), self.interpolation)
        normals = F.resize(normals, (size,size), self.interpolation)
        illum = F.resize(illum, (size,size), self.interpolation)
        return image, normals, illum

class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        self.degrees = degrees
        self.expand = expand
        self.resample = resample
        self.center = center
        self.fill = fill

    def __call__(self, image, normals, illum): 
        angle = T.RandomRotation.get_params(self.degrees)
        image =  F.rotate(image, angle, self.resample, self.expand, self.center)
        normals =  F.rotate(normals, angle, self.resample, self.expand, self.center ) 
        #TODO change normals
        illum =  F.rotate(illum, angle, self.resample, self.expand, self.center ) 
        return image, normals, illum
