# arguments: file, which song, window_start, window_finish

import librosa
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

def load_parameters():
    return yaml.load(open('params.yaml'))

params=load_parameters()

hdf5_file=h5py.File('hdf5data/' +sys.argv[1] +'.hdf5','r')

label_dataset='train_labels'
length_dataset='train_lengths'
feature_dataset='train_features'


x_data = []
y = []
my_list=[]
# one loop per batch
for j in range(1):
	#song is the index of a random song entry
    song = int(sys.argv[2])
    # = retrieves a feature from the shuffled list of songs
    feature = hdf5_file[feature_dataset][song, ...]
    # find how many samples are in this song by looking up lengths
    song_num_frames = hdf5_file[length_dataset][song, ...]
    #randomly take a section between 0 and the max available frame of a song which is as described below. Minusing one just 
    random_frame_start = int(sys.argv[3])
    random_frame_end = int(sys.argv[4])    
    # sample_excerpt must be a slice from the feature numpy
    sample_excerpt = feature[:,random_frame_start:random_frame_end]
    #convert samples into ms
    random_frame_time = (random_frame_start+int((random_frame_end-random_frame_start)/2))*params['hop_length']/params['fs']
    #iterate through label_points rows until you find an entry number that is bigger than sample_excerpt_ms
    # if previous entry is even, there are vocals (1). Else no vocals (0)
    label_points=hdf5_file[label_dataset][song, ...]
    # determine via sample location whether window has vocals or not by comparing to csv
    previous_value=-1
    for row in range(500):
        # if the row is not the last (after last row value comes zero padding)
        if label_points[row][0]>previous_value:
            if label_points[row][0]>random_frame_time:
                # third element holds the label
                label=label_points[row-1][2]
                y.append(label)
                break
            else:
                previous_value=label_points[row][0]
        else:
            label=label_points[row][2]
            y.append(label)
            break
    # I want song number, milliseconds and label in one list
    # my_list.append((song,random_frame_time,label))
    name= sys.argv[1] +', Song ' +str(song) +', WindowSize ' +str(random_frame_start) +' - ' +str(random_frame_end)
    plt.figure(figsize=(10, 4))
    plt.imshow(sample_excerpt, aspect='auto', origin='lower')
    plt.title(name)
    plt.savefig('excerptViews/' +name +'.png')
    plt.close(name)
    # print("song_index",song)
    # print("song_num_frames", song_num_frames)
    # print("random_frame_index", random_frame_index)
    # print("random_frame_time", random_frame_time)
    # print("window_frame_boundaries",random_frame_index-int(params['sample_frame_length']/2)-1,random_frame_index+int(params['sample_frame_length']/2))
    # print("sample_excerpt.shape",sample_excerpt.shape)
    # print("y", y)
    # pdb.set_trace()

# with open(sys.argv[2], "w") as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(my_list)
# csvFile.close()


# plt.figure()
# plt.plot(enum_y, y, 'bo', label='LabelList')
# plt.title('label output')
# plt.legend()
# plt.show()
