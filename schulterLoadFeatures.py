# Arguments: hdf5 file (and song list csv) name

try:
    from code.schultercore9 import *
except ImportError:
    from schultercore9 import *   # when running from terminal, the directory may not be identified as a packagefrom keras.callbacks import ModelCheckpoint
# schultercore.py has additional functions
import os, shutil, librosa
import numpy as np
import time
import matplotlib.pyplot as plt
import pdb
import csv
import math
import sys

params=load_parameters()
print(params)

print('gathering files...')
train_dir = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain/'
train_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
train_pitched_up_dir='/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain+30PercentPitch/'
train_pitched_up_files=[train_pitched_up_dir + x for x in os.listdir(train_pitched_up_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
train_pitched_down_dir='/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain-30PercentPitch/'
train_pitched_down_files=[train_pitched_down_dir + x for x in os.listdir(train_pitched_down_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
train_rate_up_dir='/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain+30PercentSr/'
train_rate_up_files=[train_rate_up_dir + x for x in os.listdir(train_rate_up_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
train_rate_down_dir='/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain-30PercentSr/'
train_rate_down_files=[train_rate_down_dir + x for x in os.listdir(train_rate_down_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
val_dir = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioVal/'
val_files = [val_dir + x for x in os.listdir(val_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
test_dir = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTest/'
test_files = [test_dir + x for x in os.listdir(test_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]

aug_files=[train_pitched_up_files,train_pitched_down_files,train_rate_up_files,train_rate_down_files]

print(len(train_files), 'train files \n', len(val_files), 'validation files \n', len(test_files), 'test files')
label_dir= '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/labels/' #soon to be changed to 'jamendo/betterlabels'
label_files = [label_dir + x for x in os.listdir(label_dir) if x.endswith('.lab')]
# for testing only
test_folder_dir= '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/testFolder/'
test_folder_files = [test_folder_dir + x for x in os.listdir(test_folder_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]

print('setting up hdf5 file...')
print('train_files',len(train_files))
print('train_pitched_up_files',len(train_pitched_up_files))
print('train_pitched_down_files',len(train_pitched_down_files))
print('train_rate_up_files',len(train_rate_up_files))
print('train_rate_down_files',len(train_rate_down_files))
num_train_instances = len(train_files)+len(train_pitched_up_files)+len(train_pitched_down_files)+len(train_rate_up_files)+len(train_rate_down_files)
print('num_train_instances', num_train_instances)
num_val_instances = len(val_files)
print('num_val_instances',num_val_instances)

image_height=params['n_mel']
image_width=int(np.round(params['max_song_length'] * params['fs'] / float(params['hop_length'])))

dataset = h5py.File('hdf5data/' +sys.argv[1] +'.hdf5', mode='w')


# we store one image per instance of size image_height x image_width holding floating point numbers
dataset.create_dataset('train_features',
                       shape=(num_train_instances, image_height, image_width+1),
                       dtype=np.float)
dataset.create_dataset('val_features',
                       shape=(num_val_instances, image_height, image_width+1),
                       dtype=np.float)
# create dataset for length of songs in ms
dataset.create_dataset('train_lengths',
                       shape=(num_train_instances, 1),
                       dtype=np.int)
dataset.create_dataset('val_lengths',
                       shape=(num_val_instances, 1),
                       dtype=np.int)
# create dataset for labellings
  # odd rows for onsets, even rows for offsets
  # saying 500 for assumed maximum annotations
dataset.create_dataset('train_labels',
                       shape=(num_train_instances, 500, 3),
                       dtype=np.float)
dataset.create_dataset('val_labels',
                       shape=(num_val_instances, 500, 3),
                       dtype=np.float)

total_num_features=-1
song_list=[]
stats_list=[]

# TRAIN TRAIN SET
print('working on train set...')

for k, audio_path in enumerate(train_files):
  total_num_features+=1
  print('  ', total_num_features, '/', num_train_instances, audio_path)
  label_name = os.path.basename(audio_path)[:-4]+'.lab'
  label_path=label_dir+label_name
  label_list=[]
  f = open(label_path,'r')
  for line in f:
    if line.find('nosing')<0:
      line=line.replace('sing','1')
    else:
      line=line.replace('nosing','0')
    line_content=line.split()
    line_ints=float(line_content[0]), float(line_content[1]), int(line_content[2])
    label_list.append(line_ints)
  f.close()
  label_array=np.asarray(label_list)
  zero_array=np.zeros((500-label_array.shape[0],label_array.shape[1]))
  label_array = np.concatenate((label_array,zero_array),axis=0)
  dataset['train_labels'][total_num_features, ...] = label_array
  feature, audio_melframe_nums = extract_feature(audio_path, params)
  
  # n=feature.shape[1]
  # m=np.mean(feature, axis=1)
  # std=np.std(feature, axis=1)
  # stats=n, m, std
  # stats_list.append(stats)

  # plot_save_feature('Train',feature, os.path.basename(audio_path))
  # we can deduce the label from the file name
  dataset['train_features'][total_num_features, ...] = feature
  dataset['train_lengths'][total_num_features, ...] = audio_melframe_nums
  song_list.append((total_num_features,os.path.basename(audio_path)))

print('working on aug set...')

for a, aug_set in enumerate(aug_files):
  for audio_path in aug_set:
    total_num_features+=1
    print('  ', total_num_features, '/', num_train_instances, audio_path)
    label_name = os.path.basename(audio_path)[:-4]+'.lab'
    label_path=label_dir+label_name
    label_list=[]
    f = open(label_path,'r')
    for line in f:
      if line.find('nosing')<0:
        line=line.replace('sing','1')
      else:
        line=line.replace('nosing','0')
      line_content=line.split()
      line_ints=float(line_content[0]), float(line_content[1]), int(line_content[2])
      label_list.append(line_ints)
    f.close()
    label_array=np.asarray(label_list)
    if a==2:
      # if aug_set is rate up by 1.3, then timestamps should be decreased multiplied by 100/130
      label_time_cols=label_array[:,0:2] 
      label_time_cols[:]=label_time_cols[:]*float(100/130)
    if a==3:
      label_time_cols=label_array[:,0:2] 
      label_time_cols[:]=label_time_cols[:]*float(100/70)
    # print(label_array)
    zero_array=np.zeros((500-label_array.shape[0],label_array.shape[1]))
    label_array = np.concatenate((label_array,zero_array),axis=0)
    dataset['train_labels'][total_num_features, ...] = label_array
    feature, audio_melframe_nums = extract_feature(audio_path, params)

    # n=feature.shape[1]
    # m=np.mean(feature, axis=1)
    # std=np.std(feature, axis=1)
    # stats=n, m, std
    # stats_list.append(stats)

    # plot_save_feature('Train',feature, os.path.basename(audio_path))
    # we can deduce the label from the file name
    dataset['train_features'][total_num_features, ...] = feature
    dataset['train_lengths'][total_num_features, ...] = audio_melframe_nums
    song_list.append((total_num_features,os.path.basename(audio_path)))

with open('saved_csvs/' +sys.argv[1] +"_songTrainH5Id.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(song_list)
csvFile.close()

n,m,std=0
# got this formula from https://math.stackexchange.com/questions/420077/find-standard-deviation-of-two-different-sets-of-numbers-when-combined
for i, stats in enumerate(stats_list):
  if i==0:
    n=stats[0]
    m=stats[1]
    std=stats[2]
  else:
    #squared_sum_of_previous_set= (mean*mean)*(numOfNums-1) + (2*mean*sum_of_previous_set) - (numOfNums*mean*mean)
    squared_sum_of_previous_set = (std*std)*(n-1) + (2*m*(n*m)) - (n*m*m)
    squared_sum_of_current_set = (stats[2]*stats[2])*(stats[0]-1) + (2*stats[1]*(stats[0]*stats[1])) - (stats[0]*stats[1]*stats[1])
    sum_of_prev_set=n*m
    sum_of_curr_set=stats[0]*stats[1]
    mean_of_both_sets=(m+stats[1])/2
    num_elements_in_both_sets=n+stats[0]
    squared_std_of_both_sets = ( (squared_sum_of_previous_set+squared_sum_of_current_set)-((2*mean_of_both_sets)*(sum_of_prev_set+sum_of_curr_set)) + (num_elements_in_both_sets*mean_of_both_sets) ) / (num_elements_in_both_sets-1)
    std_of_both_sets=math.sqrt(squared_std_of_both_sets)




print('setting 0 mean and unit variance across dataset...')


song_list=[]

# TRAIN Val SET
print('working on val set...')

for k, audio_path in enumerate(val_files):
  print('  ', k + 1, '/', len(val_files), audio_path)
  # extract the feature
  label_name = os.path.basename(audio_path)[:-4]+'.lab'
  label_path=label_dir+label_name
  label_list=[]
  f = open(label_path,'r')
  for line in f:
    if line.find('nosing')<0:
      line=line.replace('sing','1')
    else:
      line=line.replace('nosing','0')
    line_content=line.split()
    line_ints=float(line_content[0]), float(line_content[1]), int(line_content[2])
    label_list.append(line_ints)
  f.close()
  label_array=np.asarray(label_list)
  zero_array=np.zeros((500-label_array.shape[0],label_array.shape[1]))
  label_array = np.concatenate((label_array,zero_array),axis=0)
  dataset['val_labels'][k, ...] = label_array
  feature, audio_melframe_nums = extract_feature(audio_path, params)
  # plot_save_feature('Val',feature, os.path.basename(audio_path))
  # we can deduce the label from the file name
  dataset['val_features'][k, ...] = feature
  dataset['val_lengths'][k, ...] = audio_melframe_nums
  song_list.append((k,os.path.basename(audio_path)))

  with open('saved_csvs/' +sys.argv[1] +"_songValH5Id.csv", "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(song_list)
csvFile.close()
