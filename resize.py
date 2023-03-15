import torch
from PIL import Image
import torchvision.transforms as T
import os
import sys

orig_img = Image.open(sys.argv[1])
resize = T.Resize(size=[256, 256])(orig_img)
resize.save(sys.argv[2])
