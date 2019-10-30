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
import platform
import ctypes  # An included library with Python install. 
v='3.6.2'
pv=platform.python_version()
if v!=pv:
    ctypes.windll.user32.MessageBoxW(0, "Must use the 3.6.2 version of python", "", 1)
    #raise Exception("Must use the 3.6.2 version of python")
    #exit()
epoches =5
batch_size =7000
w=15
o1=15
fc=14;
percent=0.6
seq=""
if path.exists('SEQ.txt'):
      f = open('SEQ.txt')
      seq = f.readline()
      f.close()
else:
      ctypes.windll.user32.MessageBoxW(0, "The SEQ.txt file does not exist on the current directory. please check the file.", "", 1)
      #exit()
seq_len = len(seq)
content = seq[0: math.floor(seq_len*percent)]
# encoding characters to binary code
content = content.replace('A', '010')
content = content.replace('T', '011')
content = content.replace('C', '101')
content = content.replace('G', '111')
r=(len(content))%w
N=0
if r!=0: #padding to be divisible by w
  N=w-r
  seq_pad=content.ljust(N+len(content),'0')
else:
   seq_pad=content
pad_len=len(seq_pad)

def load(percent):
    
    factor = pad_len;        
    offset = 0
    part_len = factor
    data = []
    for _ in range(0, len(seq_pad), part_len):
        data.append(np.fromstring(seq_pad[offset: offset + part_len], 'u1') - ord('0'))

        offset += part_len
    
    
    a=np.array(data)

    return a

start_time = time.time()
 #the portion of data to train the auto-encoder
rows = int(math.floor(pad_len)/(w))
a = load(percent)
train_data=np.reshape(a,(rows,w))
k=math.floor((2/9)*train_data.shape[1])
if k==0:
  ctypes.windll.user32.MessageBoxW(0, "Please check the train_data array size. This array must contain 3*w columns. check your python version!", "", 1)
  #print('Please check the data_test array size. This array must contain 3*w columns. check your python version!')
  #exit()
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