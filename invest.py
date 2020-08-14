import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation

# from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime

data = pd.read_csv('samsung_data.csv')
data.head()

max_data = [data.max()]
print(max_data)

dataset = []
for i in range(data.shape[0]):
    scale = (data.iloc[i][[1, 2, 3, 4, 6]] / max_data[0][[1, 2, 3, 4, 6]])
    dataset.append(scale)    #0~1

dataset[1:5]

look_back = 20    #20
sequence_length = look_back + 1

result = []

#dataset len is 1225
for index in range(len(dataset) - sequence_length):    #0~1204
    result.append(dataset[index: index + sequence_length])    #21개씩 묶음

len(result)   #총 1204개의 자료

result = np.array(result, dtype=float)    #convert list to numpy

#print(result.shape)

row = int(round(result.shape[0] * 0.9))   #0.9정도 학습 데이터로 사용 -> 1084, 0.1은 테스트 데이터로 사용

train = result[:row, :, :]    #result의 0~1084,전체,전체  (train data 0~1084까지 분리)

np.random.shuffle(train)    #train을 섞어줌 (오버핏 방지)

x_train = train[:, :-1, :]    #(배치수,행,열)(21번 째 자료는 정답으로 쓸 것이므로 21번 째 행을 제외하고 x_train에 넣음)
# # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1, 3]    #정답은 21번째 행의 Date Open High Low Close Adj_Close Vol 중 Low로 이용

x_test = result[row:, :-1, :]    #테스트 데이터에 쓸 x_test (입력값)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1, 3]    #테스트 데이터의 답


train.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape    #미래에 까먹을 나를 위해 말하면 y리스트 값은 1차원

model = Sequential()    #층 쌓기 전 쓰는 코드(솔직히 의미는 잘 모름)

model.add(LSTM(50, return_sequences=True, input_shape=(20, 5)))    #input_shape오 입력 데이터 모양 설계 (첫 번째 층)  (20,5)

model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))    #affine

model.compile(loss='mse', optimizer='rmsprop')    #adam을 사용하면 변화를 즉각 반영하기 힘듬 따라서 rmsprop
#손실함수는 평균제곱오차 사용, optimizer은 Momentum, AdaGrad, RMSProp, Adam 중 RMSProp사용

model.summary()    #층 요약해서 보여줌

hist = model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=20,    #10 batch   20,100이 좋을듯
    epochs=100)    #12 epochs

# 손실함수
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylim(0.0, 0.005)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

pred = model.predict(x_test)

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()

#use ai
xhat = x_test[0]
prediction = model.predict(np.array([xhat]), batch_size=1)
result = prediction * max_data[0][1]
print(result)