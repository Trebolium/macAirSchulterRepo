"""
Arguments: history file path, 
"""

import sys
if sys.argv[1]=='help':
    print('Arguments: history file path')
    exit(0)


import pickle
import matplotlib.pyplot as plt
import os
import sys

model_history_file=sys.argv[1]


filename=os.path.basename(model_history_file)[:-7]
print(model_history_file)
pickle_in=open(model_history_file,'rb')
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