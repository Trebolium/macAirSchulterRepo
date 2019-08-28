import sys
if sys.argv[1]=='help':
    print('Arguments: history folder path')
    exit(0)

import pickle
import matplotlib.pyplot as plt
import os

model_history_dir=sys.argv[1]
model_history_files=[model_history_dir + x for x in os.listdir(model_history_dir) if x.endswith('.pickle') and not x.startswith('._')]


for path in model_history_files:
	filename=os.path.basename(path)[:-7]
	print(path)
	pickle_in=open(path,'rb')
	model_history=pickle.load(pickle_in)

	loss = model_history['loss']
	acc = model_history['acc']
	val_loss = model_history['val_loss']
	val_acc = model_history['val_acc']

	for x in range(len(loss)):
		print('loss: ', loss[x], 'acc: ', acc[x], 'val_loss: ', val_loss[x], 'val_acc: ', val_acc[x])

	name='Model: ' +filename +', Train and Val acc'
	epochs = range(1, len(acc) + 1)
	plt.figure()
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title(name)
	plt.legend()
	plt.savefig('modelHistory/visuals/' +filename +'_Acc.png')
	plt.close(name)

	name='Model: ' +filename +', Train and Val loss'
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title(name)
	plt.legend()
	plt.savefig('modelHistory/visuals/' +filename +'_Loss.png')
	plt.close(name)