from sklearn.model_selection import train_test_split
import tensorflow as tf
from prepare_data import *
import numpy as np
from constants import *

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)        

def conv_layer(x, ker_size, in_chan, out_chan):
    W_conv = weight_variable([ker_size, vec_size, \
                              in_chan, out_chan])
    b_conv = bias_variable([out_chan])
    h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
    h_pool = max_pool_2x2(h_conv)
    return h_pool


x = tf.placeholder(tf.float32, \
                   [None, sent_size, vec_size], name='x')
y_ = tf.placeholder(tf.float32, \
                    shape=[None, class_num], name='y_')

# reshape data to a 4d tensor
x_tensor = tf.reshape(x, [-1, sent_size, vec_size, input_chan])

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    W_conv = weight_variable([filter_size, vec_size, \
                              1, num_filters])
    b_conv = bias_variable([num_filters])
    conv = tf.nn.conv2d(x_tensor, W_conv, strides=[1, 1, 1, 1], \
                        padding="VALID")
    h = tf.nn.relu(tf.nn.bias_add(conv, b_conv))

    W_conv2 = weight_variable([filter_size, 1, 100, 200])
    b_conv2 = bias_variable([200])
    conv2 = tf.nn.conv2d(h, W_conv2, strides=[1, 1, 1, 1], \
                         padding="VALID")
    h2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))
    pooled = tf.nn.max_pool(
        h2,
        ksize=[1, sent_size - 2*filter_size + 2, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')
    pooled_outputs.append(pooled)


# Combine all the pooled features
num_filters_total = 2 * num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])


# DROPOUT
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_pool_flat, keep_prob)


# READOUT LAYER
W_fc2 = weight_variable([num_filters_total, class_num])
b_fc2 = bias_variable([class_num])


y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# def train_and_evaluate():
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, \
                                            logits=y_conv))
tf.summary.scalar('cross_entropy', cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), \
                              tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, \
                                  tf.float32), name = 'accuracy')
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

test_file = 'new_test_MR'
train_file = 'new_train_MR'
test_corpora = open(test_file, 'r')
train_corpora = open(train_file, 'r')
model_data = './saved/my_model'
new_batch = next_batch(test_corpora, 600, vec_size)

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(
        "output/train", sess.graph)
    test_writer = tf.summary.FileWriter(
        "output/test", sess.graph)
    test_new_writer = tf.summary.FileWriter(
        "output/test_new", sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(8000):
        batch = next_batch(train_corpora, 50, vec_size)
        if batch == 0:
            train_corpora.close()
            train_corpora = open(train_file, 'r')
            batch = next_batch(train_corpora, 50, vec_size)
        summary, _ = sess.run([merged, train_step], feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 0.7})
        if i % 50 == 0:
            train_writer.add_summary(summary, i)
            summary, acc_old = sess.run(
                [merged, accuracy], feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            test_writer.add_summary(summary, i)
            summary, acc_new = sess.run(
                [merged, accuracy], feed_dict={
                x: new_batch[0], y_: new_batch[1], \
                    keep_prob: 1.0})
            test_new_writer.add_summary(summary, i)
            print('step %d, acc on old %.2f, on new %.2f' % (
                i, acc_old, acc_new))
        if i % 500 == 0:
            saver.save(sess, model_data, global_step=i)
