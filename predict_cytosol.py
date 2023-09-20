#Load previously saved model
from keras.models import load_model
from tensorflow.keras.metrics import MeanIoU

model = load_model("tutorial118_mitochondria_100epochs.hdf5", compile=False)

import tensorflow as tf
import datetime

from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

images = cv2.imread('./training_data/images_cytosol/CMYC_MAX_4.tif', cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
     

image_dataset = np.array(images)

image_dataset = image_dataset /image_dataset.max()  #Can also normalize or scale using MinMax scaler


# Assuming you have a NumPy array called 'image_dataset' with shape (2808, 2688)
original_height, original_width = image_dataset.shape
tile_size = 256

# Calculate the number of tiles in both dimensions
num_tiles_height = original_height // tile_size
num_tiles_width = original_width // tile_size

# Initialize the tiled image array
tiledImage = np.zeros((num_tiles_height * num_tiles_width, tile_size, tile_size, 1), dtype=np.float32)

# Iterate over the original image and extract tiles
tile_index = 0
for i in range(num_tiles_height):
    for j in range(num_tiles_width):
        # Extract a 256x256 tile from the original image
        tile = image_dataset[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size]
        tile = np.expand_dims(tile, axis=-1)  # Add a channel dimension
        tiledImage[tile_index] = tile
        tile_index += 1



y_pred=model.predict(tiledImage)


# Assuming you have a NumPy array called 'tiledImage' with shape (x, 256, 256, 1)
num_tiles, tile_height, tile_width, num_channels = y_pred.shape
tile_size = 256

# Calculate the dimensions of the original image
original_height = num_tiles_height * tile_height
original_width = num_tiles_width * tile_width

# Initialize the original image array
reconstructed_image = np.zeros((original_height, original_width, num_channels), dtype=np.float32)

# Iterate over the tiles and reconstruct the original image
tile_index = 0
for i in range(num_tiles_height):
    for j in range(num_tiles_width):
        # Get the tile from 'tiledImage'
        tile = y_pred[tile_index]
        
        # Place the tile in the reconstructed image
        reconstructed_image[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size] = tile
        tile_index += 1


float32_array = reconstructed_image

# Assuming you have a NumPy array called 'float32_array'
# Scale the float32 array to the range [0, 255]
scaled_array = ((float32_array - float32_array.min()) / (float32_array.max() - float32_array.min()) * 255).astype(np.uint8)

# Create a monochrome OpenCV Mat
monochrome_mat = cv2.cvtColor(scaled_array, cv2.COLOR_GRAY2BGR)

# Convert the BGR image to grayscale
monochrome_mat = cv2.cvtColor(monochrome_mat, cv2.COLOR_BGR2GRAY)


# Assuming you have cv2_mat as your image
cv2.imshow("ImageNew", monochrome_mat)
cv2.waitKey(0)  # Add this to wait for a key press and then close the window
cv2.destroyAllWindows()  # Add this to close all OpenCV windows when you're done 


# Assuming you have an OpenCV Mat called 'image_mat' that you want to save as a tiled TIFF
output_file_path = "output.tiff"  # Specify the path where you want to save the tiled TIFF


success = cv2.imwrite(output_file_path, monochrome_mat)


 
