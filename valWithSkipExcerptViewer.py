# no arguments required
# goes through val songs sequentially, skipping by defined param's 'skip_size' amount
# saves spectrograms in ExcerptViews folder

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
from PIL import Image

def load_parameters():
    return yaml.load(open('params.yaml'))

params=load_parameters()
hdf5_file=h5py.File('hdf5data/withAugsWithFilterLocalNorm.hdf5','r')
num_steps = 100


frame_skips=int(params['skip_size']/(1/params['fs']*params['hop_length']))

while 1:

    for song_index in range(int(params['songs_to_validate'])):
        # print('StartLoop batch_iterator: ',batch_iterator)
        breakout=False
        song_num_frames = hdf5_file['val_lengths'][song_index, ...]
        sample_index = int(params['sample_frame_length']/2)
        # this for loop repeats after each batch is complete - hence the num_steps reference
        for i in range(num_steps):
            x_data=[]
            y=[]
            label=5
            batch_offset=i*params['batch_size']
            # print('batch_offset: ', batch_offset)
            sample_index=int(params['sample_frame_length']/2)+batch_offset
            for j in range(params['batch_size']):        
                if sample_index>=(song_num_frames-int(params['sample_frame_length']/2)-1):
                    # does this need to be here now that we have a loop that does this for us?
                    # batch_iterator=0
                    # sample_index=0
                    # song_index+=1
                    breakout=True
                    break
                else:
                    feature = hdf5_file['val_features'][song_index, ...]
                    # find how many samples are in this song by looking up lengths
                    # for k in range(int(params['sample_frame_length']/2)+1,song_num_frames-int(params['sample_frame_length']/2)-1):
                    sample_excerpt = feature[:,sample_index-int(params['sample_frame_length']/2):sample_index+int(params['sample_frame_length']/2)+1]
                    x_data.append(sample_excerpt)
                    frame_time = sample_index*params['hop_length']/params['fs']
                    label_points=hdf5_file['val_labels'][song_index, ...]

                    previous_value=-1
                    for row in range(500):
                        # if row is iterated into zero-padded territory, then we need a safety net
                        # that checks if the cuurrent row's contents (label_points[row][0]) are higher than the previous row (previous_value)
                        # the following if statement will always be true until we get to the edge of the valid label_point entries
                        if label_points[row][0]>previous_value:
                            # what if the final random frame happens to be after the last label_point?
                            # The the label+point would have to get ahead of the final frame, which would bring us to
                            # compare values of padded zeros
                            if label_points[row][0]>frame_time:
                                # go back one and get label, third element holds the label
                                label=label_points[row-1][2]
                                y.append(label)
                                # print('label: ',label)
                                break
                            else:
                                previous_value=label_points[row][0]
                        else:
                            label=label_points[row-1][2]
                            y.append(label)
                            # print('label: ',label)
                            break
                name= str(song_index) +', ' +str(frame_time) +', ' +str(label)
                plt.figure(figsize=(10, 4))
                plt.imshow(sample_excerpt, aspect='auto', origin='lower')
                plt.title(name)
                plt.savefig('excerptViews/' +name +'.png')
                plt.close(name)
                # pdb.set_trace()
                # print('sample_index, frame time: ', sample_index,frame_time)
                # print('window size: ',sample_index-int(params['sample_frame_length']/2),sample_index+int(params['sample_frame_length']/2))
                # print('excerpt shape: ',sample_excerpt.shape)
                sample_index+=frame_skips
            if breakout==True:
                break
            # x_data = np.asarray(x_data)
            # x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
            # y = np.asarray(y)
            # yield x_data, y
