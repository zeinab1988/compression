import math
import numpy as np
import matplotlib.pyplot as plt
import time
from keras import Sequential
from keras.layers import Dense
from keras import optimizers
import time
import os.path
from os import path
v='3.6.2'
if v==platform.python_version():
    raise Exception("Must be using the same versions of python")
    exit()
epoches =2
batch_size =7000
w=15
o1=15
fc=12;
seq=""
if path.exists('SEQ.txt'):
      f = open('SEQ.txt')
      seq = f.readline()
      f.close()
else:
      print('The SEQ.txt file does not exist on the current directory. please check the file.')
      exit()
seq_len = len(seq)
def load(percent):
    factor = seq_len;
    content = seq[0: math.floor(seq_len*percent)]
    # encoding characters to binary code
    content = content.replace('A', '010')
    content = content.replace('T', '011')
    content = content.replace('C', '101')
    content = content.replace('G', '111')

    offset = 0
    part_len = factor * 3
    data = []
    for _ in range(0, len(content), part_len):
        data.append(np.fromstring(content[offset: offset + part_len], 'u1') - ord('0'))

        offset += part_len
    
    
    a=np.array(data)

    #b=np.reshape(a,(rows,3*w))
    return a

start_time = time.time()
percent=0.5
rows = int(math.floor(seq_len *percent*3)/(w))
a = load(percent)
train_data=np.reshape(a,(rows,w))
k=math.floor((2/9)*train_data.shape[1])
if k==0:
  print('Please check the train_data array size. This array must contain 3*w columns. check your python version!')
  exit()
else:
# create model
  model = Sequential()
# add input layer
  model.add(Dense(units=o1, activation='relu', input_shape=(train_data.shape[1],)))
  model.add(Dense(units=fc, activation='relu'))
  model.add(Dense(units=fc, activation='relu'))
  model.add(Dense(units=fc, activation='relu'))
  x3=model.add(Dense(units=k, activation='relu'))
  model.add(Dense(units=fc, activation='relu'))
  model.add(Dense(units=fc, activation='relu'))
  model.add(Dense(units=fc, activation='relu'))
  model.add(Dense(units=train_data.shape[1], activation='sigmoid'))

  model.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy", metrics=['acc'])

  history = model.fit(x=train_data,
                    y=train_data,
                    epochs=epoches,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_split=0.2)

  model.save('auto-encoder')
print('train time', time.time() - start_time)
##print(history)
