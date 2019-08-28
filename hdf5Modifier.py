import yaml
import numpy as np
import h5py
import random
import os
import matplotlib.pyplot as plt
import math
import pdb
import sys

params=yaml.load(open('params.yaml'),Loader=yaml.FullLoader)

num_train_instances = 305
num_val_instances = 16

image_height=params['n_mel']
image_width=int(np.round(params['max_song_length'] * params['fs'] / float(params['hop_length'])))



input_hdf5=h5py.File('hdf5data/' +sys.argv[1] +'.hdf5','r')
output_hdf5 = h5py.File('hdf5data/' +sys.argv[2] +'.hdf5', mode='w')

output_hdf5.create_dataset('train_features',
                       shape=(num_train_instances, image_height, image_width+1),
                       dtype=np.float)
output_hdf5.create_dataset('val_features',
                       shape=(num_val_instances, image_height, image_width+1),
                       dtype=np.float)
# create dataset for length of songs in ms
output_hdf5.create_dataset('train_lengths',
                       shape=(num_train_instances, 1),
                       dtype=np.int)
output_hdf5.create_dataset('val_lengths',
                       shape=(num_val_instances, 1),
                       dtype=np.int)
# create dataset for labellings
  # odd rows for onsets, even rows for offsets
  # saying 500 for assumed maximum annotations
output_hdf5.create_dataset('train_labels',
                       shape=(num_train_instances, 500, 3),
                       dtype=np.float)
output_hdf5.create_dataset('val_labels',
                       shape=(num_val_instances, 500, 3),
                       dtype=np.float)

band_deets_list=[]
for mel_band in range(params['n_mel']):
	band_list=np.arange(0)
	# get std and unit variance 1 for all individual bands across all songs
	for i, song_features in enumerate(input_hdf5['train_features']):
		# add a single bands contents to a band list
		band_list=np.concatenate((band_list,song_features[mel_band]), axis=0)
		print(mel_band, ': calculating band',i, '/', len(input_hdf5['train_features']))
	band_list_mean=np.mean(band_list)
	band_list_std=np.std(band_list)
	band_deets=mel_band,band_list_mean,band_list_std
	print(band_deets)
	band_deets_list.append(band_deets)
	# scale each song to std/unitvariance
	for song_features in range(len(input_hdf5['train_features'])):
		# print(input_hdf5['train_features'][song_features][mel_band])
		# print(b)
		# a=b-band_list_mean/band_list_std
		# print(a)
		scaled_band=(input_hdf5['train_features'][song_features][mel_band]-band_list_mean)/band_list_std
		output_hdf5['train_features'][song_features,mel_band,...]=scaled_band
		print(mel_band, ': writing', song_features, '/', len(input_hdf5['train_features']))

with open('saved_csvs/' +sys.argv[1] +"database_band_mean_std.csv", "w") as csvFile:
	writer = csv.writer(csvFile)
	writer.writerows(band_deets_list)
	csvFile.close()

# discovered that hdf5 files won't replace part of a numpy. must replace all of it at once? Not even [...] will do it
# Maybe i can use for loops to save the stds and unitvars all into one list
# Can then go through each mel for each song and change them accordingly
# OOOOORRRRR it is just the fact that I should have used a comma instead of the separate bracket to fill in the data!!!
# output_hdf5['train_features'][2,...]=input_hdf5['train_features'][1,...]


# # pdb.set_trace()
# print(output_hdf5['train_features'][2])

