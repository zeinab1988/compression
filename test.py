import math
import numpy as np
import time
import os.path
from os import path
#import matplotlib.pyplot as plt
import keras
import platform
v='3.6.2'
pv=platform.python_version()
if v!=pv:
    raise Exception("Must use the 3.6.2 version of python")
    exit()

w=15
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
percent=1.0
rows = int(math.floor(seq_len *percent*3)/(w))
a = load(percent)
data_test=np.reshape(a,(rows,w))
k=math.floor((2/9)*data_test.shape[1])
if k==0:
  print('Please check the data_test array size. This array must contain 3*w columns. check your python version!')
  exit()
else:
  R=(w)/k
model = keras.models.load_model('auto-encoder')
score = model.evaluate(x=data_test, y=data_test, batch_size=20000, verbose=1)
print('compression loss:', score[0])
print('compression accuracy:', score[1])
print('compression ratio:',R)
print('compression time:', time.time() - start_time)
