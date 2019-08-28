
"""
Arguments: model name, folder/file path, percentage of song 2 analyse, csv name
Make sure the params['sample_frame_length'] value is same as it was during training
"""
import sys

if sys.argv[1]=='help':
	print('Arguments: model name, folder/file path, percentage of song 2 analyse, csv name')
	exit(0)

try:
	from code.schultercore11 import *
except ImportError:
	from schultercore11 import *   # when running from terminal, the directory may not be identified as a package
from keras.models import load_model
import csv
import time
from scipy.signal import medfilt
from tqdm import tqdm

# load model
if not os.path.isfile('models/' +sys.argv[1] +'.h5'):
	print('ERROR: No trained model found.')
	exit(0)
model = load_model('models/' +sys.argv[1] +'.h5')

parent_path=sys.argv[2]
files=[]

# load parameters
params = load_parameters()

new_dir=os.mkdir('songPredictions/'+sys.argv[4]+'/')

for r,d,f in os.walk(parent_path):
	for file in f:
		print('have a file')
		if not file.startswith('.'):
			files.append(os.path.join(r, file))


evaluation_list=[]
for f in files:
	print('File: ',f)
	audio_path=f
	# audio_path = 'jamendo/audioTest/05 - Elles disent.mp3'
	# test_files = [test_dir + x for x in os.listdir(test_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]

	timestamp_list=[]
	pred_list=[]
	round_pred_list=[]
	filtered_pred_list=[]

	# extract feature and reshape to (num_instances, image_height, image_width, num_channels)
	start_time=time.time()
	feature, audio_melframe_nums = extract_feature(audio_path, params)
	half_window=int(params['sample_frame_length']/2) # each half_window is actually half the size - 1
	max_index=audio_melframe_nums-half_window

	# 70 frames every second = (22050/315) = 14ms per frame
	# so basically just go through every frame and get a prediction
	for current_Index in tqdm(range(half_window+1,int((max_index-1)*float(int(sys.argv[3])/100)))):
		timestamp_list.append(current_Index*(params['hop_length']/params['fs']))
		# create a new subset nparray of size params['sample_frame_length']
		windowed_feature = feature[:, current_Index-(half_window+1) : current_Index+half_window]
		test_feature=windowed_feature
		#reshape for CNN model
		windowed_feature = windowed_feature.reshape((1, windowed_feature.shape[0], windowed_feature.shape[1], 1))
		# run the model on this frame
		# does the models prediction must be reffered to the 0th column of the 0th row?
		prediction=model.predict(windowed_feature)[0,0]
		pred_list.append(prediction)
		round_pred_list.append(np.round(prediction))


	#argument for medfilter is list, window_size(number of frames in 800ms as suggested by Lehner et al.[17])
	filtered_pred_list=medfilt(pred_list,55)

	roundfilteredpred_list=[]
	for entry in filtered_pred_list:
		roundfilteredpred_list.append(np.round(entry))


	all_pred_list=[]
	vocal_count=0
	for index in range(len(timestamp_list)):
		all_preds=timestamp_list[index],pred_list[index],round_pred_list[index],filtered_pred_list[index],roundfilteredpred_list[index]
		all_pred_list.append(all_preds)
		if roundfilteredpred_list[index] == 1:
			vocal_count+=1

	percentage_of_vocals=float(float(vocal_count/len(timestamp_list))*100)

	with open('songPredictions/'+sys.argv[4]+'/'+os.path.basename(f)[:-4]+'.csv', "w") as csvFile:
		writer = csv.writer(csvFile)
		header='Time','Raw_Pred','Rounded_Raw_Pred','Filtered_Pred','Round_Filtered_Pred'
		writer.writerow(header)
		for row in all_pred_list:
			writer.writerow(row)
		writer.writerow({'Percentage of song with vocals: ',str(percentage_of_vocals)})
	csvFile.close()

print('got to end')
