import os
from PIL import Image
import numpy as np
import sys

folder = sys.argv[1]
files = os.listdir(folder)
files.sort()

for file in files:

    im = Image.open(folder + '/' + file).convert('RGB')
    alpha = Image.open("./png/" + file[:-4] +'png').convert('L')
    alphagarr = np.array(alpha) 
    alphagarr[ alphagarr != 0] = 255
    alpha = Image.fromarray(np.uint8(alphagarr)).convert('L')
    # Add that alpha channel to background image
    im.putalpha(alpha)
    im.save('data/' + file[:-4] +'png')
