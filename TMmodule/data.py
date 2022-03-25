from glob2 import glob
import os
from skimage import io
import numpy as np

files = glob(os.path.join(("files", "**", "*.tif")))

# Load all files into memory to create a mean image
max_val = 2 ** 16
for idx, f in enumerate(files):
    if idx == 0:
        sum_image = io.imread(f) / max_val
    else:
        sum_image += io.imread(f) / max_val
sum_image = sum_image.astype(np.float32)
mean_image = sum_image / (idx + 1)


# In data loader, pass mean_image
# Each image becomes image - mean_image

