# from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from constants import *

class_num = 2

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# vec_size = 300
# W -- фильтр [2,300,1,100], где соотв: кол-во слов для фильтра,
# длина слова, кол-во каналов, кол-во фильтров.
# x -- вх. данные [n,16,300,1], где соотв: кол-во предл,
# число слов в предл, длина слова, каналы
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], \
                        padding='SAME', name='convolution')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], \
                          strides=[1, 2, 2, 1],  \
                          padding='SAME', name='pooling')


def conv_pool_layer(x, ker_size, in_chan, out_chan, \
               weight_name, bias_name):
    with tf.name_scope("conv_pool_layer"):
        W_conv = weight_variable([ker_size, vec_size, \
                                  in_chan, out_chan], \
                                 name=weight_name)
        b_conv = bias_variable([out_chan], name=bias_name)
        h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
    with tf.name_scope("polling_layer"):
        h_pool = max_pool_2x2(h_conv)
    return h_pool


x = tf.placeholder(tf.float32, \
                   [None, sent_size, vec_size], name='x')
y_ = tf.placeholder(tf.float32, \
                    shape=[None, class_num], name='y_')
x_tensor = tf.reshape(x, [-1, sent_size, vec_size, 1])

# THE 1 CONV LAYER
with tf.name_scope("CONV_POOL_LAYER_1"):
    ker_size1 = 2
    in_chan1 = 1
    out_chan1 = 150
    h_pool3 = conv_pool_layer(x_tensor, ker_size1, \
                         in_chan1, out_chan1, \
                         'w_1', 'b_1')

with tf.name_scope("FULLY_CONNECTED_LAYER_1"):
        # FULLY CONNECTED LAYER
    out_chan_fc = 1000
    row = h_pool3.shape[1]
    col = h_pool3.shape[2]
    depth = h_pool3.shape[3]
    in_chan_fc = (row * col * depth).value
    W_fc1 = weight_variable([in_chan_fc, out_chan_fc], 'w_fc1')
    b_fc1 = bias_variable([out_chan_fc], 'b_fc1')
    h_pool3_flat = tf.reshape(h_pool3, [-1, in_chan_fc])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1, \
                       name='relu')

with tf.name_scope("DROPOUT"):
        # DROPOUT
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='drop')
    
with tf.name_scope("READOUT_LAYER"):
    # READOUT LAYER
    W_fc2 = weight_variable([out_chan_fc, class_num], \
                            name='w_fc2')
    b_fc2 = bias_variable([class_num], 'b_fc2')
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
sess = tf.Session()
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("output", sess.graph)
writer.close()
