import torch
from PIL import Image
import torchvision.transforms as T
import os
import sys
import numpy as np

ORG_INPUT_PATH = './original_images'
MASKED_INPUT_PATH = './masked_images'
OUTPUT_PATH = './test_images'

images = os.listdir(ORG_INPUT_PATH)
images.sort()

for image in images:

    orig_img = Image.open(os.path.join(ORG_INPUT_PATH, image))
    resized_image = T.Resize(size=[256, 256])(orig_img)
    extension_lenght = len(os.path.join(ORG_INPUT_PATH, image).split('.')[-1])
    alpha = Image.open(os.path.join(MASKED_INPUT_PATH, image[:-extension_lenght] +'png') ).convert('L')
    alphagarr = np.array(alpha) 
    alphagarr[ alphagarr != 0] = 255
    alpha = Image.fromarray(np.uint8(alphagarr)).convert('L')
    resized_alpha = T.Resize(size=[256, 256])(alpha)
    resized_image.putalpha(resized_alpha)
    resized_image.save(os.path.join(OUTPUT_PATH, image[:-extension_lenght] +'png'))

print('Formated {} Images :D'.format(len(images)))
