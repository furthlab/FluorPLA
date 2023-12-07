import os
import cv2
import numpy as np

# Set the paths for the input folders
weights_org_folder = "./training_data/weights_cytosol"

# Set the paths for the output folders
weights_folder = "./training_data/cytosol/weight_maps"

# Create the output folders if they don't exist
os.makedirs(weights_folder, exist_ok=True)

# Set the ROI size and step
roi_size = (256, 256)
roi_step = 128

# Get the list of file names in the mask_org folder
file_names = [file for file in os.listdir(weights_org_folder) if file.lower().endswith(('.tif', '.tiff'))]

# Iterate over the file names
for file_name in file_names:
    # Read the mask image
    print(file_name)
    weight_path = os.path.join(weights_org_folder, file_name)
    weight = cv2.imread(weight_path, cv2.IMREAD_UNCHANGED)

    # Convert the image to float32 format for Laplacian operation
    weight = weight.astype(np.float32)
 
    # Get the dimensions of the images
    weight_height, weight_width = weight.shape[:2]

    # Iterate over the ROI positions
    for y in range(0, weight_height - roi_size[0] + 1, roi_step):
        for x in range(0, weight_width - roi_size[1] + 1, roi_step):

            # Extract the ROI from the weight
            roi_weight = weight[y:y+roi_size[0], x:x+roi_size[1]]

            # Save the ROI as a new image in the images folder
            new_weight_path = os.path.join(weights_folder, f"{file_name}_{y}_{x}.tif")
            cv2.imwrite(new_weight_path, roi_weight)
