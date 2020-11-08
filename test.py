import tensorflow as tf
import numpy as np
if __name__ == '__main__':
    BATCH_SIZE = 10
    w1 = tf.get_variable(tf.random_normal([2,3], stddev=1, seed=1))
    w2 = tf.get_variable(tf.random_normal([3,1], stddev=1, seed=1))

    x = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.placeholder(tf.float32, shape=(None, 1))

    a = tf.nn.relu(tf.matmul(x, w1))
    y_ = tf.nn.relu(tf.matmul(a, w2))

    cross_entropy = -tf.reduce_mean(y*tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    # print(1)