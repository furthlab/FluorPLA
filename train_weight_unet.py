import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

import datetime

_epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), np.float32)

# Custom loss function
def my_loss(target, output):
    return -tf.reduce_sum(target * output, len(output.shape) - 1)

# Create a weighted UNet model
def make_weighted_loss_unet(input_shape, n_classes):
    # two inputs, one for the image and one for the weight maps
    ip = L.Input(shape=input_shape)
    # the shape of the weight maps has to be such that it can be element-wise
    # multiplied to the softmax output.
    weight_ip = L.Input(shape=input_shape[:2] + (n_classes,))

    # adding the layers
    conv1 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ip)
    conv1 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = L.Dropout(0.1)(conv1)
    mpool1 = L.MaxPool2D()(conv1)

    conv2 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool1)
    conv2 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = L.Dropout(0.2)(conv2)
    mpool2 = L.MaxPool2D()(conv2)

    conv3 = L.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool2)
    conv3 = L.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = L.Dropout(0.3)(conv3)
    mpool3 = L.MaxPool2D()(conv3)

    conv4 = L.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool3)
    conv4 = L.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = L.Dropout(0.4)(conv4)
    mpool4 = L.MaxPool2D()(conv4)

    conv5 = L.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool4)
    conv5 = L.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = L.Dropout(0.5)(conv5)

    up6 = L.Conv2DTranspose(512, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv5)
    conv6 = L.Concatenate()([up6, conv4])
    conv6 = L.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = L.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = L.Dropout(0.4)(conv6)

    up7 = L.Conv2DTranspose(256, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv6)
    conv7 = L.Concatenate()([up7, conv3])
    conv7 = L.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = L.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = L.Dropout(0.3)(conv7)

    up8 = L.Conv2DTranspose(128, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv7)
    conv8 = L.Concatenate()([up8, conv2])
    conv8 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = L.Dropout(0.2)(conv8)

    up9 = L.Conv2DTranspose(64, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv8)
    conv9 = L.Concatenate()([up9, conv1])
    conv9 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = L.Dropout(0.1)(conv9)

    c10 = L.Conv2D(n_classes, 1, activation='softmax', kernel_initializer='he_normal')(conv9)

    # Add a few non trainable layers to mimic the computation of the crossentropy
    # loss, so that the actual loss function just has to peform the
    # aggregation.
    c11 = L.Lambda(lambda x: x / tf.reduce_sum(x, len(x.get_shape()) - 1, True))(c10)
    c11 = L.Lambda(lambda x: tf.clip_by_value(x, _epsilon, 1. - _epsilon))(c11)
    c11 = L.Lambda(lambda x: K.log(x))(c11)
    weighted_sm = L.multiply([c11, weight_ip])

    model = Model(inputs=[ip, weight_ip], outputs=[weighted_sm])
    return model

#Load training data
image_directory = '/training_data/cytosol/images/'
mask_directory = '/training_data/cytosol/masks/'

X = sorted(glob('training_data/cytosol/images/*.tif'))
Y = sorted(glob('training_data/cytosol/masks/*.tif'))
w = sorted(glob('training_data/cytosol/weight_maps/*.tif'))

assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

image_names = sorted(glob('training_data/cytosol/images/*.tif'))

SIZE = 256
num_images = len(X)

image_names_subset = image_names[0:num_images]
     

images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH) for img in image_names_subset]
image_dataset = np.array(images)
image_dataset = np.expand_dims(image_dataset, axis = 3)
     
mask_names = sorted(glob("training_data/cytosol/masks/*.tif"))
mask_names_subset = mask_names[0:num_images]
masks = [cv2.imread(mask, 0) for mask in mask_names_subset]

thresholded_masks = []

mask_dataset = np.array(masks)
mask_dataset = np.expand_dims(mask_dataset, axis = 3)

weight_names = sorted(glob("training_data/cytosol/weight_maps/*.tif"))
weight_names_subset = weight_names[0:num_images]
weights = [cv2.imread(weight, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH) for weight in weight_names_subset]
weight_dataset = np.array(weights)
weight_dataset = np.expand_dims(weight_dataset, axis = 3)
     
print("Image data shape is: ", image_dataset.shape)
print("Mask data shape is: ", mask_dataset.shape)
print("Max pixel value in image is: ", image_dataset.max())
print("Labels in the mask are : ", np.unique(mask_dataset))

#Normalize images
image_dataset = image_dataset /image_dataset.max()  #Can also normalize or scale using MinMax scaler
#Do not normalize masks, just rescale to 0 to 1.
mask_dataset = mask_dataset /255.  #PIxel values will be 0 or 1
#weight_dataset = weight_dataset /255.  #PIxel values will be 0 or 1

from sklearn.model_selection import train_test_split
# Split the data into training and validation sets
X_train, X_val, y_train, y_val, weight_maps_train, weight_maps_val = train_test_split(
    image_dataset, mask_dataset, weight_dataset, test_size=0.2, random_state=42)

# Define the model
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
model = make_weighted_loss_unet(input_shape, n_classes=1)

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss=my_loss, metrics=['accuracy'])
model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
batch_size = 16
epochs = 100

history = model.fit(
    [X_train, weight_maps_train],
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[tensorboard_callback],
    validation_data=([X_val, weight_maps_val], y_val)
)

# Optionally, you can save the trained model
model.save("weighted_unet_model.hdf5")
