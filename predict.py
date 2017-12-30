'''
预测 人脸验证集
'''

# ! /usr/bin/python
from deepid1 import *
import tensorflow as tf
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with tf.Session() as sess:
        saver.restore(sess, 'checkpoint/50000.ckpt')
        h1 = sess.run(h5, {h0: testX1})
        h2 = sess.run(h5, {h0: testX2})

    # 预测结果（余弦相似度（集）） consine实际上是1-余弦值
    pre_y = np.array([cosine(x, y) for x, y in zip(h1, h2)])


    # 求余弦距离阈值
    def part_mean(x, mask):
        z = x * mask
        # 一致组余弦距离总和/一致组数量
        return float(np.sum(z) / np.count_nonzero(z))


    true_mean = part_mean(pre_y, testY)  # 一致余弦距离均值
    false_mean = part_mean(pre_y, 1 - testY)  # 非一致余弦距离均值
    print(true_mean, false_mean)
    print(np.mean((pre_y < (true_mean + false_mean) / 2) == testY.astype(bool)))

    pre_y_true = []
    pre_y_false = []
    for i in range(len(testY)):
        if testY[i] == 1:
            pre_y_true.append(pre_y[i])
        else:
            pre_y_false.append(pre_y[i])

    plt.hist(pre_y_true, 50, normed=1, facecolor='g', alpha=0.75, histtype='step')
    plt.hist(pre_y_false, 50, normed=1, facecolor='r', alpha=0.75, histtype='step')
    plt.show()
