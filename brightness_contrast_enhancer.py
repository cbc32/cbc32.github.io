import numpy as np
import os
from PIL import Image, ImageEnhance

"""
After the background and shadow has been removed, to have more focus on the sign image,
it would be a good idea to increase the brightness and contrast of the image by 50%.
"""

for x in range(38):
    dir = "./data/RESIZED_DATASET/" + str(x)
    for filename in os.listdir(dir):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            original = Image.open(dir + "/" + filename)
            enhancerOne = ImageEnhance.Brightness(original)
            enhancerTwo = ImageEnhance.Contrast(original)
            im_output = enhancerOne.enhance(1.5)
            im_output = enhancerTwo.enhance(1.5)
            im_output.save(dir + "/" + filename)