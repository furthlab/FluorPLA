from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

np.random.seed(6)
lbl_cmap = random_label_cmap()

X = sorted(glob('./training_data/images_nuclei/*.tif'))
X = list(map(imread,X))

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    
model = StarDist2D(None, name='dapi', basedir='models')

img = normalize(X[0], 1,99.8, axis=axis_norm)
labels, details = model.predict_instances(img)

save_tiff_imagej_compatible('example_image.tif', img, axes='YX')
save_tiff_imagej_compatible('example_labels.tif', labels, axes='YX')
export_imagej_rois('example_rois.zip', details['coord'])
