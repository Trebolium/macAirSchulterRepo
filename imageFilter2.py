import os
import h5py
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import pdb

hdf5_path = 'hdf5data/withaugments.hdf5'
if not os.path.isfile:
    print('ERROR: HDF5-file not found! Run create_hdf5_dataset.py first!')
hdf5_file = h5py.File(hdf5_path, "r")

feature = hdf5_file['val_features'][0, ...]
sample_excerpt = feature[:,2500:2615]

# with this specific example, the numbers must be scales and normalised to these numbers which were found to be its max and min values
# sample_excerpt[:,:]=sample_excerpt[:,:]+0.2746671993100236
# sample_excerpt[:,:]=sample_excerpt[:,:]/6.762975251946389

### from here you can add to the core generator
window_size=160
highest_gauss=window_size/2
random_std=random.randint(5,7)
random_mel=random.randint(0,80)

window = signal.gaussian(window_size, std=random_std)
# The reduction of an equivalent to 10db to the numpy array was calculated by comparing pixels before and after their value underwent normalisation
# so pixels that were 10db apart - their distance was measured in the normalised version and this value was used for filtration
# window=window*-1
db_multiplier=3
# loudness filter
# plt.plot(window)
# plt.show()
# must map this to new scale

for row_index, row in enumerate(sample_excerpt):
	# print(row)
	offset=10-row_index
	for pixel_index, pixel in enumerate(row):
		# print(pixel)
		if pixel<window[int(highest_gauss+offset)]*db_multiplier:
			# pdb.set_trace()
			sample_excerpt[row_index,pixel_index]=0
		else:
			sample_excerpt[row_index,pixel_index]-=window[int(highest_gauss+offset)]*db_multiplier


print('Mel band: ',10)
# convert to image
excerpt_converted = (sample_excerpt * 255 / np.max(sample_excerpt)).astype('uint8')
excerpt_array = Image.fromarray(excerpt_converted)
# make image 3 times bigger
plt.imshow(excerpt_array)
plt.show()


