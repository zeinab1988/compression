import math
import numpy as np
import time
import os.path
from os import path
#import matplotlib.pyplot as plt
import keras
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
        data.append(np.fromstring(content[offset: offset+part_len], 'u1') - ord('0'))
        offset += part_len
    return np.array(data)
start_time = time.time()

data_test = load(1.0)
k=math.floor((2/9)*data_test.shape[1])
if k==0:
  print('Please check the data_test array size. This array must contain 3*w columns. check your python version!')
  exit()
else:
  R=(3*w)/k
model = keras.models.load_model('auto-encoder')

score = model.evaluate(data_test, data_test, verbose=2)

print('compression ratio',R)
print('compression time', time.time() - start_time)