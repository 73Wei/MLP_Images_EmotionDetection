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

# https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
print("#######1. 下載圖片##################")

path1=os.path.abspath(os.getcwd())  # 取現在的路徑
print(path1)


import shutil
import pathlib

data_dir="images/"
data_dir = pathlib.Path(data_dir)

data_dir32x32="images32x32/"
data_dir32x32 = pathlib.Path(data_dir32x32)

print("圖片的路徑：",data_dir)
print("修改後的圖片的路徑：",data_dir32x32)


print("#######2. 下載圖片的 張數 ##################")
roses=list(data_dir.glob('*/*.png'))
image_count = len(roses)
print("此路徑下面的子檔案一共有多少張png圖片",image_count)
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


import myfun
X,Y=myfun.AI_Files_LoadAllImages_Conver_SizeSave(data_dir,data_dir32x32,32,32)

print(X.shape,Y.shape)
