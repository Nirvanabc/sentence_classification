from sklearn.model_selection import train_test_split
import tensorflow as tf
from prepare_data import *
import numpy as np
from constants import *
from math import ceil

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# W -- фильтр [2,300,1,100], где соотв: кол-во слов для фильтра,
# длина слова, кол-во каналов, кол-во фильтров.
# x -- вх. данные [n,16,300,1], где соотв: кол-во предл,
# число слов в предл, длина слова, каналы
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], \
                        padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(x, ker_height, ker_width, in_chan, out_chan):
    W_conv = weight_variable([ker_height, ker_width, \
                              in_chan, out_chan])

    b_conv = bias_variable([out_chan])
    h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
    h_pool = max_pool_2x2(h_conv)
    return h_pool



class_num = 2
input_chan = 1

x = tf.placeholder(tf.float32, \
                   [None, sent_size, vec_size], name='x')
y_ = tf.placeholder(tf.float32, \
                    shape=[None, class_num], name='y_')

# reshape data to a 4d tensor
x_tensor = tf.reshape(x, [-1, sent_size, vec_size, input_chan])


# THE 1 CONV LAYER
# x = [n, 16, 300, 1]
# conv = [2, 300, 1, 50] => x = [n, 8, 150, 50]
ker_height1 = 2
ker_width1 = vec_size
in_chan1 = input_chan
out_chan1 = 100
h_pool1 = conv_layer(x_tensor, ker_height1, \
                     ker_width1, in_chan1, out_chan1)

# THE 2 CONV LAYER
# x = [n, 8, 150, 50]
# conv = [3, 150, 50, 100] => x = [n, 4, 75, 100]
ker_height2 = 3
ker_width2 = ceil(ker_width1/2)
in_chan2 = out_chan1
out_chan2 = out_chan1 * 2
h_pool3 = conv_layer(h_pool1, ker_height2, \
                     ker_width2, in_chan2, out_chan2)

# THE 3 CONV LAYER
# x = [n, 4, 75, 100]
# conv = [4, 75, 100, 200] => x = [n, 2, 38, 200] 
# ker_height3 = 4
# ker_width3 = ceil(ker_width2/2)
# in_chan3 = out_chan2
# out_chan3 = out_chan2 * 2
# h_pool4 = conv_layer(h_pool3, ker_height3, \
#                      ker_width3, in_chan3, out_chan3)


# FULLY CONNECTED LAYER
# x = [n, 2, 38, 200]
out_chan_fc = 600
row = h_pool3.shape[1]
col = h_pool3.shape[2]
depth = h_pool3.shape[3]
in_chan_fc = (row * col * depth).value
W_fc1 = weight_variable([in_chan_fc, out_chan_fc])
b_fc1 = bias_variable([out_chan_fc])
h_pool3_flat = tf.reshape(h_pool3, [-1, in_chan_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


# DROPOUT
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# READOUT LAYER
W_fc2 = weight_variable([out_chan_fc, class_num])
b_fc2 = bias_variable([class_num])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# def train_and_evaluate():
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
tf.summary.scalar('cross_entropy', cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

test_file = 'new_test_MR'
train_file = 'new_train_MR'
test_corpora = open(test_file, 'r')
train_corpora = open(train_file, 'r')
model_data = './saved/my_model'
new_batch = next_batch(test_corpora, 1000, vec_size)
new_batch = next_batch(test_corpora, 1000, vec_size)

# config = tf.ConfigProto(device_count={'CPU': 4})
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter("output/train", \
                                         sess.graph)
    test_writer = tf.summary.FileWriter("output/test", \
                                        sess.graph)
    test_new_writer = tf.summary.FileWriter("output/test_new", sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(3000):
        batch = next_batch(train_corpora, 50, vec_size)
        if batch == 0:
            train_corpora.close()
            train_corpora = open(train_file, 'r')
            batch = next_batch(train_corpora, 50, vec_size)
        summary, _ = sess.run([merged, train_step], feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 0.6})
        if i % 20 == 0:
            train_writer.add_summary(summary, i)
            summary, acc = sess.run([merged, accuracy], feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            test_writer.add_summary(summary, i)
            summary, acc = sess.run([merged, accuracy], feed_dict={
                x: new_batch[0], y_: new_batch[1], keep_prob: 1.0})
            test_new_writer.add_summary(summary, i)
            print('step %d, training accuracy %.2g' % (i, acc))
        if i % 100 == 0:
            saver.save(sess, model_data, global_step=i)
