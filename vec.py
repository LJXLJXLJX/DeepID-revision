'''
数据向量化
'''

#! /usr/bin/python
import pickle
import numpy as np
from PIL import Image

#将PIL.Image读取到的图片转成numpy.ndarray
#如果用cv2读取就不需要这一步了
def vectorize_imgs(img_path):
    with Image.open(img_path) as img:
        arr_img = np.asarray(img, dtype='float32')
        return arr_img

#读取训练集 返回两个值 第一个为图片 第二个为标签（身份编号）
def read_csv_file(csv_file):
    x, y = [], []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            path, label = line.strip().split()
            x.append(vectorize_imgs(path))
            y.append(int(label))
    return np.asarray(x, dtype='float32'), np.asarray(y, dtype='int32')

#读取成对测试集 返回三个值 前两个为图片 第三个为标签（是否为同一人）
def read_csv_pair_file(csv_file):
    x1, x2, y = [], [], []
    with open(csv_file, "r") as f:
        for line in f.readlines():
            p1, p2, label = line.strip().split()
            x1.append(vectorize_imgs(p1))
            x2.append(vectorize_imgs(p2))
            y.append(int(label))
    return np.asarray(x1, dtype='float32'), np.asarray(x2, dtype='float32'), np.asarray(y, dtype='int32')

#从pkl中读取数据
def load_data():
    with open('data/dataset.pkl', 'rb') as f:
        testX1 = pickle.load(f)
        testX2 = pickle.load(f)
        testY  = pickle.load(f)
        validX = pickle.load(f)
        validY = pickle.load(f)
        trainX = pickle.load(f)
        trainY = pickle.load(f)
        return testX1, testX2, testY, validX, validY, trainX, trainY

if __name__ == '__main__':
    testX1, testX2, testY = read_csv_pair_file('data/test_set.csv') #测试集图片对和标签（是否一致）
    validX, validY = read_csv_file('data/valid_set.csv')    #有效集
    trainX, trainY = read_csv_file('data/train_set.csv')    #训练集

    print(testX1.shape, testX2.shape, testY.shape)
    print(validX.shape, validY.shape)
    print(trainX.shape, trainY.shape)

    #将数据都整合到本地文件pkl中，方便使用
    with open('data/dataset.pkl', 'wb') as f:
        pickle.dump(testX1, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(testX2, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(testY , f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validX, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(validY, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trainX, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(trainY, f, pickle.HIGHEST_PROTOCOL)
