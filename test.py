import math
import numpy as np
import time
import matplotlib.pyplot as plt
import keras
w = 300
def load(percent):
   
    f = open('SEQ.txt')
    seq = f.readline()
    f.close()
    seq_len = len(seq)
    print(seq_len)
    total_rows = math.floor(seq_len / w)
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
k=int((2/9)*data_test.shape[1])
R=(3*w)/k
model = keras.models.load_model('auto-encoder')

score = model.evaluate(data_test, data_test, verbose=2)

print('compression ratio',R)
print('compression time', time.time() - start_time)