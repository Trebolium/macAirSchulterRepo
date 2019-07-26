# updated to have the filter augmentation
# updated to have LOCAL 0 mean and std unit variance

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


# UTILS
def load_parameters():
    """
    Load the parameter set from the yaml file and return as a python dictionary.

    Returns:
        dictionary holding parameter values
    """
    return yaml.load(open('params.yaml'))

# FEATURE EXTRACTION
def extract_feature(audio_path, params):
    audio, track_sr = librosa.load(audio_path, mono=True, sr=params['fs'])
    print('Track samplerate: ', track_sr)
    print('Track sample size: ', len(audio))
    
    # normalize
    audio /= max(abs(audio))
    max_samples = params['max_song_length'] * params['fs']  # desired max length in samples
    audio_melframe_nums = math.trunc(len(audio)/params['hop_length'])
    
    # either pad or cut to desired length
    if audio.shape[0] < max_samples:
        audio = np.pad(audio, (0, max_samples - audio.shape[0]), mode='constant')  # pad with zeros
    else:
        audio = audio[:max_samples]
    mel = librosa.feature.melspectrogram(audio,
                                             sr=params['fs'],
                                             n_mels=params['n_mel'],
                                             hop_length=params['hop_length'],
                                             n_fft=params['n_fft'], fmin=params['fmin'],fmax=params['fmax'])
    mel[mel < params['min_clip']] = params['min_clip']
    mel = librosa.amplitude_to_db(mel)
    # the line below looks like it doesnt get 0 mean and unit var by each mel line
    mel = (mel - np.mean(mel))/np.std(mel)
    print(mel.shape)
    return mel, audio_melframe_nums

def plot_save_feature(dataset, feature, fname):
    fname = fname[:-4]
    plt.figure(figsize=(10, 4))
    plt.imshow(feature, aspect='auto', origin='lower')
    plt.title(fname)
    plt.colorbar()
    dataset = 'jamendo/image' +dataset +'/'
    plt.savefig(dataset +fname +'.png')
    plt.close(fname)

# NETWORK
def generate_network(params):
    """
    Generates a keras model with convolutional, pooling and dense layers.

    Args:
        parameters: Dictionary with system parameters.

    Returns:
        keras model object.
    """
    # pre-compute "image" height and width
    # image_height = params['n_mel']
    # time to frames
    # image_width = int(np.round(params['max_song_length'] * params['fs'] / float(params['hop_length'])))

    # standard
    model = models.Sequential()
    # ***change inputs to match melspec tensors***
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(params['n_mel'], params['sample_frame_length'], 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation=params['final_activation']))


    from keras import optimizers

    # model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=params['learning_rate'], momentum=0.95, decay=0.85, nesterov=True), metrics=['acc'])
    if params['final_activation']=='softmax':
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=params['learning_rate']), metrics=['acc'])
    else:
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=params['learning_rate']), metrics=['acc'])

    return model


# iterate through the hdf5 samples


    # for each randomly take a chunk of 115 consecutive frames




def train_generator(num_steps, h5_path, params, filter_depth):
    """
    Data generator for training: Supplies the train method with features and labels taken from the hdf5 file

    Args:
        dataset: "train" or "test".
        num_steps: number of generation steps.
        shuffle: whether or not to shuffle the data
        h5_path: path to database .h5 file
        parameters: parameter dictionary

    Returns:
        feature data (x_data) and labels (y)
    """

    # def randomFilter(array,sigma):

    #     row_start_slice=random.randint(0,params['n_mel'])
    #     row_finish_slice=random.randint(row_start_slice,params['n_mel'])
    #     filter_region=array[row_start_slice:row_finish_slice]
    #     filter_region=gaussian_filter(filter_region, sigma=sigma)
    #     array[row_start_slice:row_finish_slice]=filter_region
    #     return array
    
    hdf5_file = h5py.File(h5_path, "r")  # open hdf5 file in read mode
    # point to the correct feature and label dataset
  
    while 1: # while 1 is the same is while True - an infinite loop!
        # for i in range(num_steps):  # one loop per epoch
        #     print("Numsteps: " +str(i+1) +"/" +str(num_steps))
        x_data = []
        y = []
        # one loop per batch
        for j in range(params['batch_size']):
            #random_song is the index of a random song entry
            random_song = random.randint(0,len(hdf5_file['train_labels'])-1)
            # = retrieves a feature from the shuffled list of songs
            feature = hdf5_file['train_features'][random_song, ...]
            # find how many samples are in this song by looking up lengths
            song_num_frames = hdf5_file['train_lengths'][random_song, ...]
            #randomly take a section between 0 and the max available frame of a song which is as described below. Minusing one just 
            random_frame_index = random.randint(int(params['sample_frame_length']/2)+1,song_num_frames-int(params['sample_frame_length']/2)-1)
            # sample_excerpt must be a 115 frame slice from the feature numpy
            sample_excerpt = feature[:,random_frame_index-int(params['sample_frame_length']/2)-1:random_frame_index+int(params['sample_frame_length']/2)]         
            
            window_size=160
            highest_gauss=window_size/2
            random_std=random.randint(5,7)
            random_mel=random.randint(0,79)
            # The reduction of an equivalent to 10db to the numpy array was calculated by comparing pixels before and after their value underwent normalisation
            # so pixels that were 10db apart - their distance was measured in the normalised version and this value was used for filtration
            # window=window*-1
            # 10db = 0.71
            db_multiplier=filter_depth
            window = signal.gaussian(window_size, std=random_std)

            for row_index, row in enumerate(sample_excerpt):
                # print(row)
                offset=random_mel-row_index
                for pixel_index, pixel in enumerate(row):
                    # print(pixel)
                    if pixel<window[int(highest_gauss+offset)]*db_multiplier:
                        # pdb.set_trace()
                        sample_excerpt[row_index,pixel_index]=0
                    else:
                        sample_excerpt[row_index,pixel_index]-=window[int(highest_gauss+offset)]*db_multiplier

            x_data.append(sample_excerpt)
            #convert random frame placement to ms
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

        # print('random_song: ',random_song,'sample_excerpt.shape: ',sample_excerpt.shape,)
        x_data = np.asarray(x_data)
        # print('x_data.shape',x_data.shape)
        # print(x_data.shape)
        y = np.asarray(y)
        # print(y.shape)
        # conv layers need image data format
        x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
        # pdb.set_trace()
        # send data at the end of each batch
        yield x_data, y

def val_generator(num_steps,h5_path, params):

    #convert hop size in seconds to window hop amount
    frame_skips=int(params['skip_size']/(1/params['fs']*params['hop_length']))
    hdf5_file = h5py.File(h5_path, "r")  # open hdf5 file in read mode

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
                                    y.append(label_points[row-1][2])
                                    # print('label: ',label)
                                    break
                                else:
                                    previous_value=label_points[row][0]
                            else:
                                y.append(label_points[row-1][2])
                                # print('label: ',label)
                                break
                    # pdb.set_trace()
                    # print('sample_index, frame time: ', sample_index,frame_time)
                    # print('window size: ',sample_index-int(params['sample_frame_length']/2),sample_index+int(params['sample_frame_length']/2))
                    # print('excerpt shape: ',sample_excerpt.shape)
                    sample_index+=frame_skips
                if breakout==True:
                    break
                x_data = np.asarray(x_data)
                x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
                y = np.asarray(y)
                yield x_data, y
