
"""
Arguments: model name, folder/file path, percentage of song 2 analyse, csv name
Make sure the params['sample_frame_length'] value is same as it was during training
"""
import sys
if sys.argv[1]=='help':
	print('Arguments: model name, folder/file path, percentage of song 2 analyse, csv name')
	exit(0)

try:
	from code.core import *
except ImportError:
	from core import *   # when running from terminal, the directory may not be identified as a package
from keras.models import load_model
from scipy.signal import medfilt
from tqdm import tqdm
import os, csv, time

# load model
if not os.path.isfile('models/' +sys.argv[1] +'.h5'):
	print('ERROR: No trained model found.')
	exit(0)
model = load_model('models/' +sys.argv[1] +'.h5')

parent_path=sys.argv[2]
files=[]

new_dir=os.mkdir('songPredictions/'+sys.argv[4]+'/')

for r,d,f in os.walk(parent_path):
	for file in f:
		if not file.startswith('.'):
			files.append(os.path.join(r, file))
		
for f in files:
	print('File: ',f)
	audio_path=f
	# audio_path = 'jamendo/audioTest/05 - Elles disent.mp3'
	# test_files = [test_dir + x for x in os.listdir(test_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
	# pdb.set_trace()

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

	# extract feature and reshape to (num_instances, image_height, image_width, num_channels)
	start_time=time.time()
	feature, audio_melframe_nums = extract_feature(audio_path, params)
	print("Time in seconds", time.time()-start_time)
	half_window=int(params['sample_frame_length']/2) # each half_window is actually half the size - 1
	max_index=audio_melframe_nums-half_window

	print(half_window,max_index)

	# 70 frames every second = (22050/315) = 14ms per frame
	# so basically just go through every frame and get a prediction
	for current_Index in tdqm(range(half_window+1,int((max_index-1)*float(int(sys.argv[3])/100)))):
		print('Frame Index: ',current_Index,'/',max_index)
		timestamp_list.append(current_Index*(params['hop_length']/params['fs']))
		# create a new subset nparray of size params['sample_frame_length']
		windowed_feature = feature[:, current_Index-(half_window+1) : current_Index+half_window]
		#reshape for CNN model
		windowed_feature = windowed_feature.reshape((1, windowed_feature.shape[0], windowed_feature.shape[1], 1))
		print(windowed_feature.shape)
		# run the model on this frame
		# does the models prediction must be reffered to the 0th column of the 0th row?
		prediction=model.predict(windowed_feature)[0,0]
		pred_list.append(prediction)
		round_pred_list.append(np.round(prediction))
		#round off to 0 or 1
		# list will contain frame number, prediction and rounded prediction

	#argument for medfilter is list, window_size(number of frames in 800ms as suggested by Lehner et al.[17])
	filtered_pred_list=medfilt(pred_list,55)


	roundfilteredpred_list=[]
	for entry in filtered_pred_list:
		roundfilteredpred_list.append(np.round(entry))


	all_pred_list=[]
	for index in range(len(timestamp_list)):
		all_preds=timestamp_list[index],pred_list[index],round_pred_list[index],filtered_pred_list[index],roundfilteredpred_list[index]
		all_pred_list.append(all_preds)

	label_dir='/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/labels/'

	label_name = os.path.basename(f)[:-4]+'.lab'
	label_path=label_dir+label_name
	label_points=[]
	label_file = open(label_path,'r')
	for line in label_file:
		if line.find('nosing')<0:
			line=line.replace('sing','1')
		else:
			line=line.replace('nosing','0')
		line_content=line.split()
		line_ints=float(line_content[0]), float(line_content[1]), int(line_content[2])
		label_points.append(line_ints)
	label_file.close()

	# pdb.set_trace()

	true_pos=0
	true_neg=0
	false_pos=0
	false_neg=0
	previous_time_value=0
	performance_list_name=[]
	performance_list_value=[]
	for pred_row,_ in enumerate(all_pred_list):
		for label_row,_ in enumerate(label_points):
			#if pred_time less than label_time
			if all_pred_list[pred_row][0]<label_points[label_row][0]:
				# pred value should equal label_points previous value
				if all_pred_list[pred_row][4]==label_points[label_row-1][2]:
					# and label equals 1
					if label_points[label_row-1][2]==1:
						# both are 1!
						true_pos+=1
						performance_list_name.append('true_pos')
						performance_list_value.append(5)
					else:
						# both are negative
						true_neg+=1
						performance_list_name.append('true_neg')
						performance_list_value.append(4)
					# increment pred_row
					break
				else:
					# label is 1, but pred is incorrectly 0
					if label_points[label_row-1][2]==1:
						false_neg+=1
						performance_list_name.append('false_neg')
						performance_list_value.append(2)
					else:
						# label is 0, but pred is incorrectly 1
						false_pos+=1
						performance_list_name.append('false_pos')
						performance_list_value.append(3)
					# increment pred_row
					break
			
	all_pred_perf_list=[]
	for index in range(len(timestamp_list)):
		all_preds=timestamp_list[index],pred_list[index],round_pred_list[index],filtered_pred_list[index],roundfilteredpred_list[index],performance_list_name[index],performance_list_value[index]
		all_pred_perf_list.append(all_preds)


	with open('songPredictions/'+sys.argv[4]+'/'+os.path.basename(f)[:-4]+'.csv', "w") as csvFile:
		writer = csv.writer(csvFile)
		writer.writerows(all_pred_perf_list)
	csvFile.close()

	true_poss=0
	true_negs=0
	false_poss=0
	false_negs=0
	pdb.set_trace()
	for row in all_pred_perf_list:
		if row[5]=='true_pos':
			true_poss+=1
		if row[5]=='true_neg':
			true_negs+=1
		if row[5]=='false_pos':
			false_poss+=1
		if row[5]=='false_neg':
			false_negs+=1
	print('true_poss: ',true_poss)
	print('true_negs: ',true_negs)
	print('false_poss: ',false_poss)
	print('false_negs: ',false_negs)
	precision=true_poss/(true_poss+false_poss)
	print('Precision: ', precision)
	# recall is detected_pos/all_pos
	recall=true_poss/(true_poss+false_negs)
	print('Recall: ',recall)
	specifity=true_negs/(true_poss+false_negs)
	print('Specifity: ',specifity)
	F_score=2*((precision*recall)/(precision+recall))
	print('F Score: ', F_score)