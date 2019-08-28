import librosa
import yaml
import numpy as np
import h5py
import random
import os
import matplotlib.pyplot as plt
import math
import pdb
import sys

data=h5py.File('hdf5data/' +sys.argv[1] +'.hdf5','r')

tlab='train_labels'
tlen='train_lengths'
tfea='train_features'
vlab='val_labels'
vlen='val_lengths'
vfea='val_features'


pdb.set_trace()


# print(data[tfea][int(sys.argv[2])])
	