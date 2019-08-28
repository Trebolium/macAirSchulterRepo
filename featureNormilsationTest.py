import librosa
import yaml
import numpy as np
import h5py
import random
import os
import matplotlib.pyplot as plt
import math
import pdb

params=yaml.load(open('params.yaml'),Loader=yaml.FullLoader)

audio_path='/Volumes/Samsung_T5/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain/01 - 01 Les Jardins Japonais.ogg'
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

# figure out why its not between 0 and 1!
# mel[mel < params['min_clip']] = params['min_clip']
mel_db = librosa.amplitude_to_db(mel)
mel_db_mean = np.mean(mel_db)
mel_db_std = np.std(mel_db)
mel_db_meaned = mel_db-mel_db_mean
mel_db_meaned_unitvar=mel_db_meaned/mel_db_std
pdb.set_trace()
print(mel.shape)