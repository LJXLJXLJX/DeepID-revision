'''
网络构建和训练
'''

# ! /usr/bin/python
#coding:utf-8
import numpy as np
import tensorflow as tf
import cv2
import time
from face_detection import *
from scipy.spatial.distance import cosine

# 高斯截断初始化w
def weight_variable(shape):
    with tf.name_scope('weights'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

# 全零初始化b
def bias_variable(shape):
    with tf.name_scope('biases'):
        return tf.Variable(tf.zeros(shape))

# 做Wx+b运算
def Wx_plus_b(weights, x, biases):
    with tf.name_scope('Wx_plus_b'):
        return tf.matmul(x, weights) + biases

# 神经网络层
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = Wx_plus_b(weights, input_tensor, biases)
        if act != None:
            activations = act(preactivate, name='activation')
            return activations
        else:
            return preactivate

# 卷积池化层
def conv_pool_layer(x, w_shape, b_shape, layer_name, act=tf.nn.relu, only_conv=False):
    with tf.name_scope(layer_name):
        W = weight_variable(w_shape)
        b = bias_variable(b_shape)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='conv2d')
        h = conv + b
        relu = act(h, name='relu')
        if only_conv == True:
            return relu
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max-pooling')
        return pool

#定义网络结构
with tf.name_scope('input'):
    h0 = tf.placeholder(tf.float32, [None, 55, 55, 3], name='x')
h1 = conv_pool_layer(h0, [4, 4, 3, 20], [20], 'Conv_layer_1')
h2 = conv_pool_layer(h1, [3, 3, 20, 40], [40], 'Conv_layer_2')
h3 = conv_pool_layer(h2, [3, 3, 40, 60], [60], 'Conv_layer_3')
h4 = conv_pool_layer(h3, [2, 2, 60, 80], [80], 'Conv_layer_4', only_conv=True)
# Deepid层，与最后两个卷积层相连接
with tf.name_scope('DeepID1'):
    h3r = tf.reshape(h3, [-1, 5 * 5 * 60])
    h4r = tf.reshape(h4, [-1, 4 * 4 * 80])
    W1 = weight_variable([5 * 5 * 60, 160])
    W2 = weight_variable([4 * 4 * 80, 160])
    b = bias_variable([160])
    h = tf.matmul(h3r, W1) + tf.matmul(h4r, W2) + b
    h5 = tf.nn.relu(h)


saver = tf.train.Saver()

def compare(pic1,pic2):

    pic1=faceDetect(pic1).reshape(1,55,55,3)
    pic2=faceDetect(pic2).reshape(1,55,55,3)


    cv2.imshow('1', pic1.reshape(55,55,3))
    cv2.imshow('2', pic2.reshape(55,55,3))

    with tf.Session() as sess:
        saver.restore(sess, 'checkpoint/50000.ckpt')
        h1 = sess.run(h5, {h0: pic1})
        h2 = sess.run(h5, {h0: pic2})
        return np.array([cosine(x, y) for x, y in zip(h1, h2)])[0]

pic1='data/crop_images_DB/Kevin_Keegan/0/aligned_detect_0.180.jpg'
pic2='data/crop_images_DB/Raghad_Saddam_Hussein/1/aligned_detect_1.1432.jpg'
start=time.time()
sim=compare(pic1,pic2)
print(sim,' ',sim<=0.5,'\n',time.time()-start,'s')


cv2.waitKey(0)
cv2.destroyAllWindows()