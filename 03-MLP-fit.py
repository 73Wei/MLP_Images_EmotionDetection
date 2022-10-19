#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "WCH"
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#from tensorflow.examples.tutorials.mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds
#from tensorflow.examples.tutorials.mnist

# https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
print("#######1. 下載圖片##################")

path1=os.path.abspath(os.getcwd())  # 取現在的路徑
print(path1)


import shutil
import pathlib

data_dir="images32x32/"
data_dir = pathlib.Path(data_dir)


print("圖片的路徑：",data_dir)


print("#######2. 下載圖片的 張數 ##################")
roses=list(data_dir.glob('*/*.png'))
image_count = len(roses)
print("此路徑下面的子檔案一共有多少張jpg圖片",image_count)
#roses = list(data_dir.glob('roses/*'))

print("#######2. 下載圖片的顯示  ##################")
import matplotlib.pyplot as plt
newsize = (32, 32)
t1=PIL.Image.open(str(roses[0]))         # 讀檔案
t1=t1.resize(newsize)                    # 調整圖片大小
t1=np.asarray(t1)                        # 資料轉成numpy
plt.imshow(t1)
#plt.show()

print("#######3. 圖片轉 numpy處理 ##################")

def AI_Files_LoadAllImages(IMAGEPATH):
    IMAGEPATH=str(IMAGEPATH)
    #IMAGEPATH = 'images'
    dirs = os.listdir(IMAGEPATH)
    X = []
    Y = []
    print(dirs)
    i = 0    # 分類答案
    for name in dirs:      # 每一個路徑
        # check if folder or file
        t1=IMAGEPATH + "/" + name
        if os.path.exists(t1) and  os.path.isdir(t1):
            file_paths = glob.glob(os.path.join(IMAGEPATH + "\\" + name, '*.*'))    # 找底下的*.* 的檔案
            for path3 in file_paths:     # 處理每一張圖片
                path3=path3.replace("\\","/")
                try:
                    im_rgb = np.asarray(PIL.Image.open(str(path3)).resize(newsize))
                    X.append(im_rgb)
                    Y.append(i)
                    print("Y=",i," 讀取：",path3)
                except:
                    print("不是圖片檔案或無法開啟：",str(path3))
            i = i + 1

    X = np.asarray(X)     # list 轉 Numpy
    Y = np.asarray(Y)
    return X,Y

X,Y=AI_Files_LoadAllImages(data_dir)
print(X.shape,Y.shape)

print("#######4. numpy處理 ##################")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.05)
print('x_train = ' + str(x_train.shape))
print('y_train = ' + str(y_train.shape))

print("####### 5. 顯示資料內容 ##################")
# 顯示資料內容
def printMatrixE(a):
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      str1=""
      for j in range(0,cols):
         str1=str1+("%3.0f " % a[i, j])
      print(str1)
   print("")

printMatrixE(x_train[1])

# 顯示其中的圖形
num=1
plt.title('x_train[%d]  Label: %d' % (num, y_train[num]))
#plt.imshow(x_train[num], cmap=plt.get_cmap('gray_r'))
plt.imshow(x_train[num], cmap=plt.get_cmap('gray'))   # 真正的灰階樣子
#plt.show()

# 影像的類別數目
num_classes = 6
# 輸入的手寫影像解析度
img_rows, img_cols = 32, 32

print('x_train before reshape:', x_train.shape)
# 將原始資料轉為正確的影像排列方式
dim=img_rows*img_cols*1
x_train = x_train.reshape(x_train.shape[0], dim)
x_test = x_test.reshape(x_test.shape[0], dim)
print('x_train after reshape:', x_train.shape)


# 標準化輸入資料
print('x_train before div 255:',x_train[0][180:195])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train before div 255 ', x_train[0][180:195])

print('y_train shape:', y_train.shape)
print(y_train[:10])
# 將數字轉為 One-hot 向量
category=6
y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)
print("y_train2 to_categorical shape=",y_train2.shape)     #輸出 (60000, 10)
print(y_train2[:10])


# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu,
    input_dim=dim))  # 784=28*28
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
    loss = tf.keras.losses.categorical_crossentropy,
    metrics = ['accuracy'])
# 設定模型的 Loss 函數、Optimizer 以及用來判斷模型好壞的依據（metrics）


# 顯示模型
model.summary()
"""

lastCost=1.0
for step in range(20000):
    history = model.fit(x_train, y_train2)       # 進行訓練的因和果的資料
          #batch_size=100,                 # 設定每次訓練的筆數
          #epochs=2000,                     # 設定訓練的次數，也就是機器學習的次數
          #verbose=1)  #
    if step % 10 == 0:
        W, b = model.layers[0].get_weights()
        print("step", step, " Weights = ", W, ", bias =", b, " train cost", history)
        #  如果答案比較好，就儲存下來
        if (history[0] < lastCost):
            lastCost = history[0]
            # 保存模型權重
            model.save_weights("model.h5")
        #  如果 loss cost  就提早離開
        if (lastCost < 0.00000001):
            break

"""
# 訓練模型
history=model.fit(x_train, y_train2,       # 進行訓練的因和果的資料
          batch_size=100,                 # 設定每次訓練的筆數
          epochs=5000,                     # 設定訓練的次數，也就是機器學習的次數
          verbose=1)

#測試
score = model.evaluate(x_test, y_test2)                        # 計算測試正確率
print("score:",score)                                          # 輸出測試正確率
predict = model.predict(x_test)                                # 取得每一個結果的機率
print("Ans:",np.argmax(predict,axis=-1))


#predict2 = model.predict_classes(x_test[:10])                  # 取得預測答案2
#print("predict_classes:",predict2[:10])

#輸出預測答案2
print("y_test",y_test[:10])                                     # 實際測試的果

#保存模型架構
with open("model.json", "w") as json_file:
   json_file.write(model.to_json())
#保存模型權重
model.save_weights("model.h5")








