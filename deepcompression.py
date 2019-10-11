# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 22:09:38 2019

@author: nazemi
"""



import keras
from keras.datasets import mnist
from keras import Sequential
from keras import layers
from keras import optimizers
from keras import Model
import matplotlib.pyplot as plt
import numpy as np

"""DATA_SETTING"""

file_path = "chrm19.txt"
file_data = open(file_path,"r")
line = file_data.readline()
X_train =[]
while line:
  line.replace('Y','')
  line.replace('y','')
  line.replace('m','')
  line.replace('M','')
  line.replace('r','')
  line.replace('R','')
  line.replace('K','')
  line.replace('k','')
  line.replace('s','')
  line.replace('S','')
  line.replace('w','')
  line.replace('W','')
  line.replace('a','100')
  line.replace('t','011')
  line.replace('c','001')
  line.replace('g','111')
  line.replace('n','000')
  line.replace('A','100')
  line.replace('T',' 011')
  line.replace('C','001')
  line.replace('G','111')
  line.replace('N','000')
  if len(line)==181:
    temp=[]
    for j in range(181):
      if line[j]=="1":
        temp.append(1)
      elif line[j]=="0" :
        temp.append(0)
    X_train.append(temp)
  line = file_data.readline()

file_data.close()
X_train = np.array(X_train)

"""# AUTO_ENCODER_PART"""

#creat autoencoder model

input_layer= layers.Input(shape=(X_train.shape[1],))

#x1 = layers.Dense(170,activation="relu")(input_layer)
x1 = layers.Dense(100,activation="relu")(input_layer)
x2 = layers.Dense(100,activation="relu")(x1)
x3 = layers.Dense(100,activation="relu")(x2)
encoder_layer = layers.Dense(36,activation="relu",name="encoder")(x3)
x1 = layers.Dense(100,activation="relu")(encoder_layer)
x2 = layers.Dense(100,activation="relu")(x1)
x3 = layers.Dense(100,activation="relu")(x2)
#x1 = layers.Dense(170,activation="relu")(x2)
decoder_layer = layers.Dense(X_train.shape[1],activation="sigmoid")(x3)
autoencoder=Model(input_layer, decoder_layer)

# train auto encoder model
autoencoder.summary()
autoencoder.compile(optimizer=optimizers.Adam(),loss="binary_crossentropy",metrics=['acc'])
hsitory = autoencoder.fit(x=X_train,
                    y=X_train,
                    epochs=50,
                    batch_size=1,
                    shuffle=True,
                    validation_split=0.20)
acc = hsitory.history['acc']
val_acc = hsitory.history['val_acc']
loss = hsitory.history['loss']
val_loss = hsitory.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'mo', label='Training_acc')
plt.plot(epochs, val_acc, 'y', label='Validation_acc')
plt.title('acc with encoder')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'mo', label='Training_loss')
plt.plot(epochs, val_loss, 'y', label='Validation_loss')
plt.title('loss with encoding')
plt.legend()
plt.show()
file_path2 = "h.txt"
file_data2 = open(file_path2,"r")
line2 = file_data2.readline()
X_test =[]
while line2:
  if len(line2)==181:
    temp2=[]
    for j in range(181):
      if line2[j]=="1":
        temp2.append(1)
      elif line2[j]=="0" :
        temp2.append(0)
    X_test.append(temp)
  line2 = file_data2.readline()

file_data2.close()
X_test = np.array(X_test)

X2 = X_test[1].reshape(1,-1)
X3 = autoencoder.predict(X2)

for k in range (180):
    if X3[0][k]<=0.2:
        X3[0][k]=0;
    else:
        X3[0][k]=1;
X4=(X3-X2);  
j=0
for k in range (180):
    if X4[0][k]!=0:
        j=j+1;

accu=100-((j/179)*100)        
# print(y_train[0].argmax())
# plt.imshow(X.reshape(28,28))
# plt.title("encode and decode image of digit 5")
# plt.show()
# plt.figure()
# plt.imshow(x_train[0].reshape(28,28))
# plt.title("orginale image of digit 5")
# plt.show()

encoder_model = Model(input_layer, encoder_layer)
x_train = encoder_model.predict(X_train)
# X_test = encoder_model.predict(x_test)


# # make classification model
# print(X_train.shape)
# model_with_encodeing = Sequential()


# model_with_encodeing.add(layers.Dense(15,activation="tanh",input_shape=(X_train.shape[1],)))
# model_with_encodeing.add(layers.Dense(10,activation="softmax"))


# model_with_encodeing.summary()
# model_with_encodeing.compile(optimizer=optimizers.SGD(),loss="binary_crossentropy",metrics=['acc'])
# history = model_with_encodeing.fit(x=X_train,
#                                 y=y_train,
#                                 epochs=110,
#                                 batch_size=300,
#                                 shuffle=True,
#                                 validation_data=(X_test,y_test))

"""encoder results"""

#acc = history.history['acc']
#val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'mo', label='Training_acc')
# plt.plot(epochs, val_acc, 'y', label='Validation_acc')
# plt.title('acc with encoder')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'mo', label='Training_loss')
# plt.plot(epochs, val_loss, 'y', label='Validation_loss')
# plt.title('loss with encoding')
# plt.legend()

# plt.show()