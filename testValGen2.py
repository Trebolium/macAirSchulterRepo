# import librosa
import yaml
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
# from keras import optimizers, models, layers
import h5py
import matplotlib.pyplot as plt
import math
import pdb
import csv

def load_parameters():
    return yaml.load(open('params.yaml'), Loader=yaml.FullLoader)

params=load_parameters();

hdf5_file = h5py.File('hdf5data/basic.hdf5', "r")  # open hdf5 file in read mode
# point to the correct feature and label dataset
batch_iterator=0
song_index=2
song_num_frames = hdf5_file['val_lengths'][song_index, ...]
x_data = []
y = []

while sample_index<(song_num_frames-int(params['sample_frame_length']/2)-1):
    # one loop per batch
    print('newwhileloop')
    sample_index=int(params['sample_frame_length']/2)
    # from this loop onwards is the batch
    for j in range(params['batch_size']):        
        if sample_index>=(song_num_frames-int(params['sample_frame_length']/2)-1):
            batch_iterator=0
            song_index+=1
            break
        else:
            offset=batch_iterator*params['batch_size']
            # = retrieves a feature from the shuffled list of songs
            feature = hdf5_file['val_features'][song_index, ...]
            # find how many samples are in this song by looking up lengths
            # for k in range(int(params['sample_frame_length']/2)+1,song_num_frames-int(params['sample_frame_length']/2)-1):
            sample_excerpt = feature[:,sample_index+offset-int(params['sample_frame_length']/2):sample_index+offset+int(params['sample_frame_length']/2)+1]
            # x_data.append(sample_excerpt)
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
                        label=(sample_index,frame_time,label_points[row-1][2])
                        y.append(label)
                        print('label: ',label)
                        break
                    else:
                        previous_value=label_points[row][0]
                else:
                    label=(sample_index,frame_time,label_points[row-1][2])
                    y.append(label)
                    print('label: ',label)
                    break

        # x_data = np.asarray(x_data)
        # y = np.asarray(y)
        # pdb.set_trace()
        print('frame_time: ',frame_time)
        print('sample_index: ', sample_index)
        print('window size: ',sample_index+offset-int(params['sample_frame_length']/2),sample_index+offset+int(params['sample_frame_length']/2))
        print('excerpt shape: ',sample_excerpt.shape)
        sample_index+=1
    batch_iterator +=1


        # song_index +=1
        # yield y
        # print(sample_index,y)

# params=load_parameters();
# a=val_generator('hdf5data/basic.hdf5', params)
# y_list=[]

# hdf5_file = h5py.File('hdf5data/basic.hdf5', "r")  # open hdf5 file in read mode

# a=val_generator('hdf5data/basic.hdf5', params)
# pdb.set_trace()
# for l in range(int(hdf5_file['val_lengths'][0])):
#     y_list.append(a)

with open('test.csv', "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(y)
csvFile.close()

