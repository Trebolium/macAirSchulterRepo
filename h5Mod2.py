import yaml
import numpy as np
import h5py
import random
import os
import matplotlib.pyplot as plt
import math
import pdb
import sys
import csv

params=yaml.load(open('params.yaml'),Loader=yaml.FullLoader)

h5=h5py.File('hdf5data/' +sys.argv[1] +'.hdf5','r+')
# h5=h5py.File('hdf5data/' +sys.argv[2] +'.hdf5', mode='w')

# h5=h5
# print('First', h5['train_features'][0,:,0:115])
# h5['train_features'][0,:,0:115]=h5['train_features'][0,:,0:115]*3
# print('Second', h5['train_features'][0,:,0:115])


band_deets_list=[]
for mel_band in range(params['n_mel']):
	band_list=np.arange(0)
	# get std and unit variance 1 for all individual bands across all songs
	for i, song_features in enumerate(h5['train_features']):
		# add a single bands contents to a band list
		band_list=np.concatenate((band_list,song_features[mel_band]), axis=0)
		print(mel_band, ': gathering band',i, '/', len(h5['train_features']))
	band_list_mean=np.mean(band_list)
	band_list_std=np.std(band_list)
	band_deets=mel_band,band_list_mean,band_list_std
	print(band_deets)
	band_deets_list.append(band_deets)
	# scale each song to std/unitvariance
	for song_features in range(len(h5['train_features'])):
		# print(h5['train_features'][song_features][mel_band])
		# print(b)
		# a=b-band_list_mean/band_list_std
		# print(a)
		scaled_band=(h5['train_features'][song_features][mel_band]-band_list_mean)/band_list_std
		h5['train_features'][song_features,mel_band,...]=scaled_band
		print(mel_band, ': writing band', song_features, '/', len(h5['train_features']))

with open('saved_csvs/' +sys.argv[1] +"database_band_mean_std.csv", "w") as csvFile:
	writer = csv.writer(csvFile)
	writer.writerows(band_deets_list)
	csvFile.close()
