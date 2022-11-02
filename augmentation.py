import numpy as np
import os
from PIL import Image, ImageEnhance



"""
if we've removed the background, maybe the model shoud ignore color?
The code adds 10 images for each image. 5 rotations for unflipped and left-right flipped versions.
"""
images_by_class = dict()
classes = 37
for c in classes:
    images = np.array()
    dir = "data\\RESIZED_DATASET\\" + c
    for filename in os.listdir(dir):
        if filename.endswith(".jpg"):
            original = Image.open(dir)
            enhancer = ImageEnhance.Color(original)
            bw = enhancer.enhance(factor=0.0)
            bw_flipped = bw.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for image in [bw, bw_flipped]:
                for angle in [-30, -15, 0, 15, 30]:
                    i = image.rotate(angle)
                    images = np.concatenate([images, np.asarray(i)])
    images_by_class[c] = images