# requires arguments: hdf5 file, model name, db filter depth

try:
    from code.schultercore11 import *
except ImportError:
    from schultercore11 import *   # when running from terminal, the directory may not be identified as a package
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import sys
import time
import pdb

start_time=time.time()
print('Ignore this test print' +str(time.time()-start_time))

params=load_parameters()

#just to get number of files per set, used further down in step computing
# print('gathering files...')
# train_dir = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTrain/'
# train_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
# val_dir = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioVal/'
# val_files = [val_dir + x for x in os.listdir(val_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
# test_dir = '/Users/brendanoconnor/Desktop/APP/MXX-git2-/SchulterReproduction/jamendo/audioTest/'
# test_files = [test_dir + x for x in os.listdir(test_dir) if x.endswith('.wav') or x.endswith('.ogg') or x.endswith('.mp3')]
# print(len(train_files), 'train files \n', len(val_files), 'validation files \n', len(test_files), 'test files')


# make sure we have the hdf5 data file
hdf5_path = 'hdf5data/' +sys.argv[1] +'.hdf5'
if not os.path.isfile:
    print('ERROR: HDF5-file not found! Run create_hdf5_dataset.py first!')
    exit(0)

# load parameters
params = load_parameters()

# generate CNN model
model = generate_network(params)
model.summary()  # print a summary of the model

# pre-compute number of steps
hdf5_file = h5py.File(hdf5_path, "r")
# num_val_steps = params['num_train_steps'] * len(val_files)/len(train_files)
num_val_steps=0
for i in range(params['songs_to_validate']):
	num_val_steps = num_val_steps + int(hdf5_file['val_lengths'][i])/params['batch_size']

num_val_steps=int(num_val_steps)
# callbacks
# save the best performing model
save_best = ModelCheckpoint('models/' +sys.argv[2] +'.h5', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=params['patience'], verbose=0, mode='auto', baseline=None, restore_best_weights=False)
x_val, y_val=load_linear_val_data(hdf5_path, params)

# train
# train_datagen= ImageDataGenerator(
#     # rescale=1./255,
#     # rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     # shear_range=0.2,
#     # zoom_range=0.2,
#     # horizontal_flip=True
#     fill_mode='nearest')

# val_datagen = ImageDataGenerator(rescale=1./255)

# # train_generator ouputs batches of data
# # train_datagen receives these batches and augments them
# train_input = train_datagen.train_generator(params['num_steps'],hdf5_path, params)
# val_input = val_datagen.val_generator(params['num_steps'],hdf5_path, params)

#fit_generator receives these augmented images and fits them to model

history = model.fit_generator(train_generator(params['num_train_steps'], hdf5_path, params, float(sys.argv[3])),
                    steps_per_epoch=params['num_train_steps'],
                    epochs=params['epochs'],
                    validation_data=(x_val, y_val),
                    # number of steps not needed when there is no generator
                    # validation_steps=num_val_steps,
                    callbacks=[save_best,early_stop])

print(params)
print('Training took: ' +str(time.time()-start_time) +' seconds.')

make_history(history, sys.argv[2])

# save model history to disk as the same name as model

