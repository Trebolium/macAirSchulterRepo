"""

Arguments: (str)path, (int)samplerate

converts all audio files to wav files at 22050
takes from a specific directory and outputs to that same one

required arguments: samplerate

"""

import librosa
import sys
import os

parent_path=sys.argv[1]

files=[]

for r,d,f in os.walk(parent_path):
	for file in f:
		if not file.startswith('.'):
			files.append(os.path.join(r, file))
		
for f in files:
	print('coverting: ',f)
	audio, track_sr = librosa.audio.load(f, sr=int(sys.argv[2]), mono=True)
	librosa.output.write_wav(f[:-4]+'.wav', audio, track_sr, norm=False)
