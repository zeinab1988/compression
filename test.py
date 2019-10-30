import math
import numpy as np
import time
import os.path
from os import path
#import matplotlib.pyplot as plt
import keras
import platform
import ctypes  # An included library with Python install.   
#ctypes.windll.user32.MessageBoxW(0, "'The SEQ.txt file does not exist on the current directory. please check the file.'", "", 1)
v='3.6.2'
pv=platform.python_version()
if v!=pv:
    ctypes.windll.user32.MessageBoxW(0, "Must use the 3.6.2 version of python", "", 1)
    #raise Exception("Must use the 3.6.2 version of python")
    #exit()
w=15
percent=1.0
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

    #b=np.reshape(a,(rows,3*w))
    return a

start_time = time.time()
 #the portion of data to train the auto-encoder
rows = int(math.floor(pad_len)/(w))
a = load(percent)
data_test=np.reshape(a,(rows,w))
k=math.floor((2/9)*data_test.shape[1])
if k==0:
    ctypes.windll.user32.MessageBoxW(0, "Please check the data_test array size. This array must contain 3*w columns. check your python version!", "", 1)
  #print('Please check the data_test array size. This array must contain 3*w columns. check your python version!')
  #exit()
else:
  R=(w)/k
model = keras.models.load_model('auto-encoder')
score = model.evaluate(x=data_test, y=data_test, batch_size=5000, verbose=1)
print('compression loss:', score[0])
print('compression accuracy:', score[1])
print('compression ratio:',R)
print('compression time:', time.time() - start_time)