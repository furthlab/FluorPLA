import cv2
import numpy as np

from skimage.segmentation import find_boundaries

mask_image = cv2.imread('./training_data/masks_cytosol/CMYC_MAX_4.tif', cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

# Convert the image to float32 format for Laplacian operation
mask_image = mask_image.astype(np.float32)

# Apply Laplacian filter
B = cv2.Laplacian(mask_image, cv2.CV_32F)

# Take the absolute value in-place
B = np.abs(B)

# Divide by B (this will produce a divide by zero warning that you can safely ignore)
B /= B

B = np.nan_to_num(B, nan=0)

# Multiply by mask_image
B *= mask_image

C = mask_image - B

# Convert B back to uint8 if needed
C = C.astype(np.uint8)

# Define the kernel (structuring element) for erosion
kernel_size = 3  # Adjust the size as needed
kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    

eroded_image = cv2.erode(C, kernel, iterations=8)

# Define the new width and height (25% of the original size)
new_width = int(eroded_image.shape[1] * 1)
new_height = int(eroded_image.shape[0] * 1)

# Resize the image using bicubic interpolation
eroded_image = cv2.resize(eroded_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


_, thresholded_mask = cv2.threshold(eroded_image, 0, 255, cv2.THRESH_BINARY)
_, thresholded_mask2 = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY)


# Display the result using imshow
cv2.imshow('Result Image', thresholded_mask)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

boundary = thresholded_mask2-thresholded_mask
boundary = boundary.astype(np.uint8)

dist = cv2.distanceTransform(255-thresholded_mask, cv2.DIST_L2, cv2.DIST_MASK_5)


dist = cv2.distanceTransform(255-boundary, cv2.DIST_L2, 5)

dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

border_loss_map = w0 * np.exp((-1 * (dist) ** 2) / (2 * (sigma ** 2)))
border_loss_map= border_loss_map.astype(np.uint8)
cv2.imwrite("border_loss_map.tif", border_loss_map)

cv2.imshow('Distance Trans', dist)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("border_loss_map.tif", border_loss_map)

# https://jaidevd.com/posts/weighted-loss-functions-for-instance-segmentation/

#generate weight map
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
    xSum = np.subtract.outer(X2, X1) ** 2
    xSum = np.subtract.outer(X2, X1) ** 2

    ySum = (Y2[:, np.newaxis] - Y1) ** 2

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

weight = make_weight_map(thresholded_mask)

eroded_image = cv2.resize(eroded_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
