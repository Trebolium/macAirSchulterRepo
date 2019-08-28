from scipy.ndimage import gaussian_filter
import sys
import os
import yaml
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random

def randomFilter(array,sigma):

	row_start_slice=random.randint(0,params['n_mel'])
	print(row_start_slice)
	row_finish_slice=random.randint(row_start_slice,row_start_slice+random.randint(1,20))
	print(row_finish_slice)
	filter_region=array[row_start_slice:row_finish_slice]
	filter_region=gaussian_filter(filter_region, sigma=sigma)
	array[row_start_slice:row_finish_slice]=filter_region
	return array

params=yaml.load(open('params.yaml'),Loader=yaml.FullLoader)
sigma=1

hdf5_path = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/hdf5data/' +'basic' +'.hdf5'
if not os.path.isfile:
    print('ERROR: HDF5-file not found! Run create_hdf5_dataset.py first!')
hdf5_file = h5py.File(hdf5_path, "r")
feature = hdf5_file['val_features'][0, ...]
song_length= hdf5_file['val_lengths'][0, ...]
window_start=random.randint(int((params['sample_frame_length']/2)+1),int(song_length-(params['sample_frame_length']/2)-1))
window_finish=window_start+params['sample_frame_length']
print(window_start,window_finish)
array = feature[:,window_start:window_finish]
print(array.shape)

plt.figure()
imgplot = plt.imshow(array)
plt.show()

array=randomFilter(array,sigma)

plt.figure()
imgplot = plt.imshow(array)
plt.show()

