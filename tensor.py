import tensorflow as tf
from prepare_data import my_dictionary
import numpy as np
from random import shuffle

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# W -- фильтр [2,300,1,100], где соотв: кол-во слов для фильтра,
# длина слова, кол-во каналов, кол-во фильтров.
# x -- вх. данные [n,15,300,1], где соотв: кол-во предл,
# число слов в предл, длина слова, каналы
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], \
                        padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# bad, good, vec_size = result()
# data = bad + good
# shuffle(data)

sent_size = 16
class_num = 2

x = tf.placeholder(tf.float32, [None, sent_size, vec_size, 1])
y_ = tf.placeholder(tf.float32, shape=[None, class_num])
  

# THE 1 CONV LAYER
ker_size1 = 2
in_chan1 = 1
out_chan1 = 100

# x = [16, 300, 1]
# conv = [2, 300, 1, 50] 
W_conv1 = weight_variable([ker_size1, vec_size, \
                           in_chan1, out_chan1])

b_conv1 = bias_variable([out_chan1])

# use h_conv1._shape to see the result shape
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# THE 2 CONV LAYER
# x = [8, 150, 50]
# conv = [3, 150, 50, 100]
ker_size2 = 3
in_chan2 = out_chan1
out_chan2 = out_chan1 * 2
W_conv2 = weight_variable([ker_size2, vec_size, \
                           in_chan2, out_chan2])

b_conv2 = bias_variable([out_chan2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# THE 3 CONV LAYER
# x = [4, 75, 100]
# conv = [4, 75, 100, 200] 
ker_size3 = 4
in_chan3 = out_chan2
out_chan3 = out_chan2 * 2
W_conv3 = weight_variable([ker_size3, vec_size, \
                           in_chan3, out_chan3])

b_conv3 = bias_variable([out_chan3])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


# FULLY CONNECTED LAYER
# x = [2, 38, 200, 400] 
out_chan_fc = 40000
row = h_pool3.shape[1]
col = h_pool3.shape[2]
depth = h_pool3.shape[3]
in_chan_fc = (row * col * depth).value
W_fc1 = weight_variable([in_chan_fc, out_chan_fc])
b_fc1 = bias_variable([out_chan_fc])
h_pool3_flat = tf.reshape(h_pool3, [-1, in_chan_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


# DROPOUT
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# READOUT LAYER
W_fc2 = weight_variable([out_chan_fc, class_num])
b_fc2 = bias_variable([class_num])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

def train_and_evaluate():
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    batch = data
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
                
