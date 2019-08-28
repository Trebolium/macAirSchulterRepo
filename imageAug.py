import keras
import tensorflow as tf
import os
import h5py
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

hdf5_path = '/Volumes/Samsung_T5/APP/MXX-git2-/SchulterReproduction/hdf5data/withAugsFromWavLocalBlockNorm.hdf5'
if not os.path.isfile:
    print('ERROR: HDF5-file not found! Run create_hdf5_dataset.py first!')
hdf5_file = h5py.File(hdf5_path, "r")

# grab slice or numpy arrays from dataset
feature = hdf5_file['val_features'][0, ...]
sample_excerpt = feature[:,1400:1515]

scales = list(np.arange(0.8, 1.0, 0.01))
# make a list for box dimensions
boxes = np.zeros((len(scales), 4))

# fill the boxes array with sequentially changing  dimensions
for i, scale in enumerate(scales):
    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    boxes[i] = [x1, y1, x2, y2]

sample_reshape = sample_excerpt.reshape((sample_excerpt.shape[0], sample_excerpt.shape[1],1))
converted = tf.convert_to_tensor(sample_reshape)
converted.shape
crops = tf.image.crop_and_resize([converted], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(80, 115))
print(crops[0].shape)
# crops = tf.image.crop_and_resize([sample_reshape], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(80, 115))
# tensor is supposed to be dense? I thought this wanted a n image format
# does tf.image.crop_and_resize want sample_reshape to be in a different format, if so which one?
plt.figure(1)
reshaped_crop=crops[0][:,:,0]
# crops[0] is a tensor object. How to convert it to an array again?
# tf.convertIMageTensorToArray(crops[0])

# TypeError: Image data of dtype object cannot be converted to float
imgplot=plt.imshow()