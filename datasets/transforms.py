import numpy as np
from PIL import Image
import random
import mxnet as mx 
from opencv_transforms import functional as F
from opencv_transforms import transforms as T
from albumentations import functional as FA

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)

        return image
        
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        
        image = F.center_crop(image, self.size)
        return image

class Resize(object): 
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = (size,size) #TODO according to what is size
        self.interpolation = interpolation

    def __call__(self, image): #TODO warning for mask -> nearest interpolation?
        image = F.resize(image, self.size, self.interpolation)
        return image

class ToTensor(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        return image

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
        return image

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
        return image

class Random90DegRotClockWise(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            angle = 90
            image =  F.rotate(image, angle, resample=False, expand=False, center=None)
            
        return image

class Random180DegRot(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            angle = 180
            image =  F.rotate(image, angle, resample=False, expand=False, center=None)
        return image

class Albumentations(object):
    def __init__(self, hue_limit, sat_limit, flip_prob):
        self.hue_limit = hue_limit
        self.sat_limit = sat_limit
        self.flip_prob=flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            
            hue_shift=random.uniform(-self.hue_limit, self.hue_limit)
            sat_shift=random.uniform(-self.sat_limit, self.sat_limit)
            image[:,:,:3] = FA.shift_hsv(image[:,:,:3], hue_shift=hue_shift,sat_shift=sat_shift, val_shift=0)

        return image

class Albumentations2(object):
    def __init__(self, hue_limit, sat_limit, flip_prob):
        self.hue_limit = hue_limit
        self.sat_limit = sat_limit
        self.flip_prob=flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            
            mx_ex_int_array = mx.nd.array(image[:,:,:3] / 255.)
            aug = mx.ndarray.image.random_saturation(mx_ex_int_array[:,:,:3],0.,1.)
            aug = mx.ndarray.image.random_hue(aug,0.,1.)
            image[:,:,:3] = (aug.asnumpy() * 255).astype(np.uint8)
        return image


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, ):

        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        return image

class RandomResize(object):
    def __init__(self, low, high, interpolation=Image.BILINEAR):
        self.low = low
        self.high = high
        self.interpolation = interpolation

    def __call__(self, image, ): 
        size = np.random.randint(self.low, self.high)
        image = F.resize(image, (size,size), self.interpolation)
        return image

class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        self.degrees = degrees
        self.expand = expand
        self.resample = resample
        self.center = center
        self.fill = fill

    def __call__(self, image): 
        angle = T.RandomRotation.get_params(self.degrees)
        image =  F.rotate(image, angle, self.resample, self.expand, self.center)

        return image

