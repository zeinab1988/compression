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
epoches =1
batch_size = 200
w = 700;
def load(percent):
    seq=""
    if path.exists('SEQ.txt'):
      f = open('SEQ.txt')
      seq = f.readline()
      f.close()
    else:
      print('The SEQ.txt file does not exist on the current directory. please check the file.')
      exit()
    seq_len = len(seq)
    print(seq_len)
    if (w!=0 and seq_len!=0):
      total_rows = math.floor(seq_len / w)
    else:
      print('the sequence was not correctly loaded and w must be greater than zero')
      exit()
    content = seq[0: int(w * math.floor(total_rows * percent))]
    # encoding characters to binary code
    content = content.replace('A', '100')
    content = content.replace('T', '011')
    content = content.replace('C', '001')
    content = content.replace('G', '111')

    offset = 0
    part_len = w * 3
    data = []
    for _ in range(0, len(content), part_len):
        data.append(np.fromstring(content[offset: offset + part_len], 'u1') - ord('0'))

        offset += part_len

    return np.array(data)

start_time = time.time()
train_data = load(.6)
k=math.floor((2/9)*train_data.shape[1])
if k==0:
  print('Please check the train_data array size. This array must contain 3*w columns. check your python version!')
  exit()
else:
# create model
  model = Sequential()
# add input layer
  model.add(Dense(units=300, activation='elu', input_shape=(train_data.shape[1],)))
  model.add(Dense(units=200, activation='elu'))
  model.add(Dense(units=200, activation='elu'))
  model.add(Dense(units=200, activation='elu'))
  model.add(Dense(units=k, activation='elu'))
  model.add(Dense(units=200, activation='elu'))
  model.add(Dense(units=200, activation='elu'))
  model.add(Dense(units=200, activation='elu'))
  model.add(Dense(units=200, activation='elu'))
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
#print(history)