"""
Arguments: model name, csv name, percentage of song to analyze
"""

try:
    from code.schultercore10 import *
except ImportError:
    from schultercore10 import *   # when running from terminal, the directory may not be identified as a package
from keras.models import load_model
import os
import numpy as np
import csv
import sys
import time
from scipy.signal import medfilt

# test on one song, as we will be assigning values to every window
audio_path = 'jamendo/audioTest/05 - Elles disent.mp3'
# test_files = [test_dir + x for x in os.listdir(test_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]

timestamp_list=[]
pred_list=[]
round_pred_list=[]
filtered_pred_list=[]

# load parameters
params = load_parameters()

# for k, audiofile in enumerate(test_files):
# 	print('  ', k + 1, '/', len(test_files), audiofile)
#     feature, audio_melframe_nums = extract_feature(audiofile, params)
#     feature_list.append(feature)


# load model
if not os.path.isfile('models/' +sys.argv[1] +'.h5'):
    print('ERROR: No trained model found.')
    exit(0)
model = load_model('models/' +sys.argv[1] +'.h5')

# extract feature and reshape to (num_instances, image_height, image_width, num_channels)
start_time=time.time()
feature, audio_melframe_nums = extract_feature(audio_path, params)
print("Time in seconds", time.time()-start_time)
start_Index=int(params['sample_frame_length']/2)+1
finish_Index=feature.shape[1]-int(params['sample_frame_length']/2)-1

print(start_Index,finish_Index)

# for every frame, sliding across the chosen song
# chose what portion out of 100% of the song to be analyzed
for current_Index in range(start_Index,start_Index+int((finish_Index-start_Index)*float(int(sys.argv[3])/100))):
	print('Frame Index: ',current_Index,'/',finish_Index)
	timestamp_list.append(current_Index*params['hop_length']/params['fs'])
	# create a new subset nparray of size params['sample_frame_length']
	windowed_feature = feature[:,current_Index-int(params['sample_frame_length']/2)-1:current_Index+int(params['sample_frame_length']/2)]
	#reshape for CNN model
	windowed_feature = windowed_feature.reshape((1, windowed_feature.shape[0], windowed_feature.shape[1], 1))
	# run the model on this frame
	prediction=model.predict(windowed_feature)[0,0]
	pred_list.append(prediction)
	round_pred_list.append(np.round(prediction))
	#round off to 0 or 1
	# list will contain frame number, prediction and rounded prediction

filtered_pred_list=medfilt(round_pred_list,5)

# with open('songPredictions/FilteredPrediction_' +sys.argv[2] +'.csv', "w") as csvFile:
#     writer = csv.writer(csvFile)
#     for value in filtered_pred_list:
#     	# pdb.set_trace()
#     	writer.writerow(value)
# csvFile.close()

all_predictions_list=[]
for index in range(len(timestamp_list)):
	time_pred_roundpred=timestamp_list[index],pred_list[index],round_pred_list[index],filtered_pred_list[index]
	all_predictions_list.append(time_pred_roundpred)

with open('songPredictions/' +sys.argv[2] +'.csv', "w") as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(all_predictions_list)
csvFile.close()

pdb.set_trace()


