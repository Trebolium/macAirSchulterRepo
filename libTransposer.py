import librosa
import yaml
import random
import sys


params=yaml.load(open('params.yaml',),Loader=yaml.FullLoader)

random_transpose=random.randint(-7,7)

src_audio='jamendo/audioTest/05 - Elles disent.mp3'
dst_audio='jamendo/testAudioCrap/' +sys.argv[1] +'TransposedBy' +str(random_transpose) +'.wav'

print(random_transpose)

audio, track_sr = librosa.load(src_audio, mono=True, sr=params['fs'])
audio_transposed = librosa.effects.pitch_shift(y=audio,sr=params['fs'],n_steps=random_transpose)
librosa.output.write_wav(path=dst_audio, y=audio_transposed, sr=params['fs'])