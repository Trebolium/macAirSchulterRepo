import librosa
import yaml
import random
import sys

params=yaml.load(open('params.yaml'),Loader=yaml.FullLoader)

random_stretch=random.uniform(0.7,1.3)


src_audio='jamendo/audioTest/05 - Elles disent.mp3'
dst_audio='jamendo/testAudioCrap/' +sys.argv[1] +'StretchedBy' +str(round(random_stretch,2)) +'.wav'

print(random_stretch)

audio, track_sr = librosa.load(src_audio, mono=True, sr=params['fs'])
audio_stretched = librosa.effects.time_stretch(y=audio, rate=random_stretch)
librosa.output.write_wav(path=dst_audio, y=audio_stretched, sr=params['fs'])