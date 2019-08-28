import librosa
import yaml
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import optimizers, models, layers
import h5py
import random
import os
import matplotlib.pyplot as plt
import math
import pdb
from scipy import signal
from scipy.ndimage import gaussian_filter

params=yaml.load(open('params.yaml'),Loader=yaml.FullLoader)

hdf5_path = 'hdf5data/withaugments.hdf5'
if not os.path.isfile:
    print('ERROR: HDF5-file not found! Run create_hdf5_dataset.py first!')
hdf5_file = h5py.File(hdf5_path, "r")
 
while 1: # while 1 is the same is while True - an infinite loop!
    x_data = []
    y = []
    # one loop per batch
    for j in range(params['batch_size']):
        #random_song is the index of a random song entry
        random_song = 266
        # random_song = random.randint(0,len(hdf5_file['train_labels'])-1)
        # = retrieves a feature from the shuffled list of songs
        feature = hdf5_file['train_features'][random_song, ...]
        # find how many samples are in this song by looking up lengths
        song_num_frames = hdf5_file['train_lengths'][random_song, ...]
        #randomly take a section between 0 and the max available frame of a song which is as described below. Minusing one just 
        random_frame_index = random.randint(int(params['sample_frame_length']/2)+1,song_num_frames-int(params['sample_frame_length']/2)-1)
        print('song_num_frames',song_num_frames)
        print('random_frame_index',random_frame_index)
        # sample_excerpt must be a 115 frame slice from the feature numpy
        sample_excerpt = feature[:,random_frame_index-int(params['sample_frame_length']/2)-1:random_frame_index+int(params['sample_frame_length']/2)]         
        
        ### FILTER AUGMENTATION SECTION
        # window_size=160
        # highest_gauss=window_size/2
        # random_std=random.randint(5,7)
        # random_mel=random.randint(0,79)
        # # The reduction of an equivalent to 10db to the numpy array was calculated by comparing pixels before and after their value underwent normalisation
        # # so pixels that were 10db apart - their distance was measured in the normalised version and this value was used for filtration
        # # window=window*-1
        # db_multiplier=0.71
        # window = signal.gaussian(window_size, std=random_std)
        # # print('random_std', random_std)        
        # # print('random_mel', random_mel)

        # for row_index, row in enumerate(sample_excerpt):
        #     # print(row)
        #     offset=random_mel-row_index
        #     # print('window index', int(highest_gauss+offset))
        #     for pixel_index, pixel in enumerate(row):
        #         if pixel<window[int(highest_gauss+offset)]*db_multiplier:
        #             # pdb.set_trace()
        #             sample_excerpt[row_index,pixel_index]=0
        #         else:
        #             sample_excerpt[row_index,pixel_index]-=window[int(highest_gauss+offset)]*db_multiplier

        ### SAVE SAMPLE_EXCERPT TO ITS BATCH
        x_data.append(sample_excerpt)
        #convert random frame placement to ms
        print('random_song: ',random_song)
        print('sample_excerpt.shape: ',sample_excerpt.shape)
        ### FIND Y VALUE BY SEARCHING THE LABEL DATASET FOR THE RELEVANT TIMESTAMP
        random_frame_time = random_frame_index*params['hop_length']/params['fs']
        #iterate through label_points rows until you find an entry number that is bigger than sample_excerpt_ms
        # if previous entry is even, there are vocals (1). Else no vocals (0)
        label_points=hdf5_file['train_labels'][random_song, ...]
        # determine via sample location whether window has vocals or not by comparing to csv
        previous_value=-1
        for row in range(500):
            # if the row is not the last (after last row value comes zero padding)
            if label_points[row][0]>previous_value:
                # if label exceeds time instance of random frame
                if label_points[row][0]>random_frame_time:
                    # go back one and get label, third element holds the label
                    label=label_points[row-1][2]
                    y.append(label)
                    break
                else:
                    previous_value=label_points[row][0]
            else:
                label=label_points[row-1][2]
                y.append(label)
                break

    # print('sample_excerpt.shape: ',sample_excerpt.shape, 'len(x_data)',len(x_data))
    # for feature in x_data:
    #     print('feature.shape',feature.shape)
    # print(x_data)
    x_data = np.asarray(x_data)
    print('x_data.shape',x_data.shape)
    # print(x_data.shape)
    y = np.asarray(y)
    # print(y.shape)
    # conv layers need image data format
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
    # pdb.set_trace()