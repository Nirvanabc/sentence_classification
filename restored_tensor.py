import tensorflow as tf
from prepare_data import *

sent_size = 16
class_num = 2
vec_size = 100

saver = tf.train.import_meta_graph('saved/my_model-2080.meta')
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint("./saved/"))

graph = tf.get_default_graph()

x = graph.get_tensor_by_name("x:0")
y_ = graph.get_tensor_by_name("y_:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
accuracy = graph.get_tensor_by_name("accuracy:0")
corpora_file = 'corpora_MR'
corpora = open(corpora_file, 'r')


def batch_classifier_online():
    batch = next_batch(corpora, 50, vec_size)
    # the next line is mistakable! It sets all variables
    # to random values!
    # sess.run(tf.global_variables_initializer())
    feed_dict ={x:batch[0], y_:batch[1], keep_prob:1.0}
    with sess.as_default():
        print (sess.run(accuracy, feed_dict))


def sent_classifier_online(sent, label):
    sent = [sent.split()]
    sent = prepare_corpora(sent, vec_size, sent_size)
    feed_dict = {x:sent, y_: [label], keep_prob:1.0}
    with sess.as_default():
        print (sess.run(accuracy, feed_dict))
    
