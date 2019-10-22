import math
import numpy as np
import matplotlib.pyplot as plt
import time
from keras import Sequential
from keras.layers import Dense
from keras import optimizers

epoches =1
batch_size = 100
w = 300;
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
        data.append(np.fromstring(content[offset: offset + part_len], 'u1') - ord('0'))

        offset += part_len

    return np.array(data)

start_time = time.time()
train_data = load(.6)
k=train_data.shape[1]
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