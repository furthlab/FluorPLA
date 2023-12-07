import os
import cv2
import numpy as np

from skimage.segmentation import find_boundaries

w0 = 10
sigma = 5

def make_weight_map(mask):
    """
    Generate the weight map for a binary mask as specified in the UNet paper.
    
    Parameters
    ----------
    mask: array-like
        A 2D binary mask of shape (image_height, image_width).

    Returns
    -------
    array-like
        A 2D weight map of shape (image_height, image_width).
    
    """
    nrows, ncols = mask.shape
    mask = (mask > 0).astype(int)
    
    # Compute the distance map
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    bounds = find_boundaries(mask, mode='inner')
    X2, Y2 = np.nonzero(bounds)
    xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
    ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
    distMap = np.sqrt(xSum + ySum)
    
    # Calculate the border loss map
    w0 = 10.0  # Adjust this value as needed
    sigma = 5.0  # Adjust this value as needed
    border_loss_map = w0 * np.exp((-1 * (distMap) ** 2) / (2 * (sigma ** 2)))
    
    # Compute the class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - mask.sum() / loss.size
    w_0 = 1 - w_1
    loss[mask == 1] = w_1
    loss[mask == 0] = w_0
    
    # Combine border loss and class weight to get the final weight map
    weight_map = border_loss_map + loss
    
    return weight_map

# Set the paths for the input folders
mask_org_folder = "./training_data/masks_cytosol"
images_org_folder = "./training_data/images_cytosol"

# Set the paths for the output folders
mask_folder = "./training_data/cytosol/masks"
images_folder = "./training_data/cytosol/images"
weights_folder = "./training_data/cytosol/weight_maps"

# Create the output folders if they don't exist
os.makedirs(mask_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)
os.makedirs(weights_folder, exist_ok=True)

# Set the ROI size and step
roi_size = (256, 256)
roi_step = 128

# Get the list of file names in the mask_org folder
file_names = [file for file in os.listdir(mask_org_folder) if file.lower().endswith(('.tif', '.tiff'))]

# Iterate over the file names
for file_name in file_names:
    # Read the mask image
    mask_path = os.path.join(mask_org_folder, file_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to float32 format for Laplacian operation
    mask = mask.astype(np.float32)
    
    # Apply Laplacian filter
    B = cv2.Laplacian(mask, cv2.CV_32F)
    
    # Take the absolute value in-place
    B = np.abs(B)
    
    # Divide by B (this will produce a divide by zero warning that you can safely ignore)
    B /= B
    B = np.nan_to_num(B, nan=0)
    
    # Multiply by mask
    B *= mask
    
    mask = mask - B
    
    # Convert B back to uint8 if needed
    mask = mask.astype(np.uint8)

    # Define the kernel (structuring element) for erosion
    kernel_size = 3  # Adjust the size as needed
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    # Perform erosion
    mask = cv2.erode(mask, kernel, iterations=4)    
    # 
    # # Define the new width and height (25% of the original size)
    # new_width = int(mask.shape[1] * 0.25)
    # new_height = int(mask.shape[0] * 0.25)
    # 
    # # Resize the image using bicubic interpolation
    # mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # 
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    _, thresholded_mask2 = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY)
    
    
    boundary = thresholded_mask2-mask
    boundary = boundary.astype(np.uint8)
    
    dist = cv2.distanceTransform(255-boundary, cv2.DIST_L2, cv2.DIST_MASK_5)

    weight_map = w0 * np.exp((-1 * (dist) ** 2) / (2 * (sigma ** 2)))
    weight_map = weight_map.astype(np.uint8)    

    # Read the corresponding input image
    image_path = os.path.join(images_org_folder, file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    #image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Get the dimensions of the images
    mask_height, mask_width = mask.shape[:2]
    image_height, image_width = image.shape[:2]

    # Iterate over the ROI positions
    for y in range(0, mask_height - roi_size[0] + 1, roi_step):
        for x in range(0, mask_width - roi_size[1] + 1, roi_step):
            # Extract the ROI from the mask
            roi_mask = mask[y:y+roi_size[0], x:x+roi_size[1]]

            # Extract the ROI from the image
            roi_image = image[y:y+roi_size[0], x:x+roi_size[1]]
            
            # Extract the ROI from the weight
            roi_weight = weight_map[y:y+roi_size[0], x:x+roi_size[1]]
            
            # Save the ROI as a new image in the mask folder
            new_mask_path = os.path.join(mask_folder, f"{file_name}_{y}_{x}.tif")
            cv2.imwrite(new_mask_path, roi_mask)

            # Save the ROI as a new image in the images folder
            new_image_path = os.path.join(images_folder, f"{file_name}_{y}_{x}.tif")
            cv2.imwrite(new_image_path, roi_image)

            # Save the ROI as a new image in the images folder
            new_weight_path = os.path.join(weights_folder, f"{file_name}_{y}_{x}.tif")
            cv2.imwrite(new_weight_path, roi_weight)
