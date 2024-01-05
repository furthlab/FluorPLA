from __future__ import print_function, unicode_literals, absolute_import, division
import argparse

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

#Load previously saved model
from keras.models import load_model
from tensorflow.keras.metrics import MeanIoU

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

import pandas as pd

def calculate_padded_dimensions(original_height, original_width, tile_size=256):
    padded_height = (original_height + tile_size - 1) // tile_size * tile_size
    padded_width = (original_width + tile_size - 1) // tile_size * tile_size
    return padded_height, padded_width

#path = '/Users/danielfurth/Documents/GitHub/FluorPLA/figure01/230913/MAX_CMYC_MAX_1_MMStack_Pos0.ome.tif'

def main(image_path):
  model = load_model("cytosol_unet_400epochs.hdf5", compile=False)

  # Get the basename of the image file
  image_basename = os.path.basename(image_path)

  # Remove the file extension and add '.csv' as the new extension
  csv_file = os.path.splitext(image_basename)[0] + '.csv'

  # List to store the loaded image
  multiP = []
  
  ret, multiP = cv2.imreadmulti(image_path, flags=cv2.IMREAD_ANYDEPTH)
  
  # Check if images were loaded successfully
  if ret:
      # Separate the channels
      antibody = multiP[0] # multiP[2]   # First channel
      nuclei = multiP[1]  # Second channel
      cytosol =  multiP[2]  # multiP[0] # Third channel
      
  # Create a mask for values below the threshold
  maskTmp = (antibody < 150) # 400
  
  # Set values below the threshold to 0
  antibody[maskTmp] = 0

  
  image_dataset = np.array(cytosol)
  
  image_dataset = image_dataset /image_dataset.max()  #Can also normalize or scale using MinMax scaler
  
  
  # Assuming you have a NumPy array called 'image_dataset' with shape (2808, 2688)
  original_height, original_width = image_dataset.shape
  tile_size = 256
  
  padded_height, padded_width = calculate_padded_dimensions(original_height, original_width)
  image_dataset_padded = np.pad(image_dataset, ((0, padded_height - original_height), (0, padded_width - original_width)), mode='constant')
  
  # Calculate the number of tiles in both dimensions
  num_tiles_height = padded_height // tile_size
  num_tiles_width = padded_width // tile_size
  
  # Initialize the tiled image array
  tiledImage = np.zeros((num_tiles_height * num_tiles_width, tile_size, tile_size, 1), dtype=np.float32)
  
  # Iterate over the original image and extract tiles
  tile_index = 0
  for i in range(num_tiles_height):
      for j in range(num_tiles_width):
          # Extract a 256x256 tile from the original image
          tile = image_dataset_padded[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size]
          tile = np.expand_dims(tile, axis=-1)  # Add a channel dimension
          tiledImage[tile_index] = tile
          tile_index += 1
  
  
  y_pred=model.predict(tiledImage)
  
  # Assuming you have a NumPy array called 'tiledImage' with shape (x, 256, 256, 1)
  num_tiles, tile_height, tile_width, num_channels = y_pred.shape
  tile_size = 256
  
  # Calculate the dimensions of the original image
  new_height = num_tiles_height * tile_height
  new_width = num_tiles_width * tile_width
  
  # Initialize the original image array
  reconstructed_image = np.zeros((new_height, new_width, num_channels), dtype=np.float32)
  
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
  
  monochrome_mat = monochrome_mat[:original_height, :original_width]
  
  # Apply a Gaussian filter with a 3-pixel sigma
  sigma = 3
  filtered_image = cv2.GaussianBlur(monochrome_mat, (0, 0), sigma)
  
  # Threshold the filtered image to create a binary image
  _, binary_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY)
  
  # Perform image segmentation to create a labeled image
  num_labels, labeled_image = cv2.connectedComponents(binary_image)
  
  # Note: 'labeled_image' now contains the labeled regions. Each region has a unique integer label.
  # 'num_labels' represents the total number of labeled regions.
  
  # You can display the labeled image using a colormap for visualization:
  #colored_labeled_image = cv2.applyColorMap((labeled_image * 255 / num_labels).astype(np.uint8), cv2.COLORMAP_JET)
  
  
  axis_norm = (0,1)   # normalize channels independently
  # axis_norm = (0,1,2) # normalize channels jointly
  
  model = StarDist2D(None, name='dapi', basedir='models')
  
  img = normalize(nuclei, 1,99.8, axis=axis_norm)
  labels, details = model.predict_instances(img)
  
  # Get unique labels in the "labels" image
  unique_labels = np.unique(labels)
  
  # Create a DataFrame to store label intensities
  data = {"Label": [], "Parent": [], "Cyto": [], "Background": [], "Nuclei": []}
  
  # Iterate through unique labels and calculate mean intensity
  for label in unique_labels:
      if label == 0:
          continue  # Skip background label (0)
  
      # Create a mask for the current label
      label_mask = (labels == label).astype(np.int8)
      
  
      # Calculate the mean intensity within the mask in the "antibody" image
      label_intensity = np.mean(antibody[label_mask == 1])
      parent = int(np.mean(labeled_image[label_mask == 1]))
      
      cytomask = (labeled_image == parent).astype(np.int8)
      label_mask2 = cytomask - label_mask
      background = np.mean(antibody[labeled_image == 0])
      
      cyto_intensity = np.mean(antibody[label_mask2 == 1])
  
      # Append the data to the DataFrame
      data["Label"].append(label)
      data["Parent"].append(parent)
      data["Background"].append(background)
      data["Cyto"].append(cyto_intensity)
      data["Nuclei"].append(label_intensity)
  
  
  # Create a DataFrame
  df = pd.DataFrame(data)
  
  # Save the results to a CSV file
  #csv_file = "nuclei_intensities.csv"
  df.to_csv(csv_file, index=False)
  
  print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing script")
    parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()
    main(args.image_path)
