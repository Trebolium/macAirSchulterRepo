import librosa
import yaml
import random
import sys
import os

params=yaml.load(open('params.yaml'),Loader=yaml.FullLoader)

random_transpose=random.randint(-7,7)

# random_stretch=random.uniform(0.7,1.3)



train_dir = 'jamendo/audioTrain/'
train_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]

#3.6 was choosen because that is 30% of an octave, which I consider to be a 100% transposition
aug_transpositions=[-3.6,3.6]
aug_sr=[0.7,1.3]


for k, audio_path in enumerate(train_files):
	print('Converting song ' +str(k+1) +'/' +str(len(train_files)))
	audio, track_sr = librosa.load(audio_path, mono=True, sr=params['fs'])
	name=label_name = os.path.basename(audio_path)
	for m, aug_strength in enumerate(aug_transpositions):
		audio_transposed = librosa.effects.pitch_shift(y=audio,sr=params['fs'],n_steps=aug_strength)
		if m == 0:
			dst_audio='jamendo/audioTrain-30PercentPitch/'+name
		else:
			dst_audio='jamendo/audioTrain+30PercentPitch/'+name
		print(dst_audio)
		librosa.output.write_wav(path=dst_audio, y=audio_transposed, sr=params['fs'])
		print('converted ' +name +'and sent to ' +dst_audio)
	for m, aug_strength in enumerate(aug_sr):
		audio_stretched = librosa.effects.time_stretch(y=audio, rate=aug_strength)
		if m == 0:
			dst_audio='jamendo/audioTrain-30PercentSr/'+name
		else:
			dst_audio='jamendo/audioTrain+30PercentSr/'+name
		librosa.output.write_wav(path=dst_audio, y=audio_stretched, sr=params['fs'])
		print('converted ' +name +'and sent to ' +dst_audio)


# print(random_transpose)

# audio_transposed = librosa.effects.pitch_shift(y=audio,sr=params['fs'],n_steps=random_transpose)
# librosa.output.write_wav(path=dst_audio, y=audio_transposed, sr=params['fs'])

# print(random_stretch)

# audio, track_sr = librosa.load(audio_path, mono=True, sr=params['fs'])
# librosa.output.write_wav(path=dst_audio, y=audio_stretched, sr=params['fs'])