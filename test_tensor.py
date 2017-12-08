import tensorflow as tf
from test_preproc import my_dictionary
import numpy as np
from random import shuffle
print ("!!!!")

# no output
bad, good, vec_size = my_dictionary()
data = [[1,2,3,4], [1,1,1,1], [2,2,2,2,]]
sent_size = 2
class_num = 2
vec_size = 2
data = tf.reshape(data, [-1, sent_size, vec_size, 1])




# MUCH output!
bad, good, vec_size = my_dictionary()
data = bad + good
sent_size = 2
class_num = 2
vec_size = 2
data = tf.reshape(data, [-1, sent_size, vec_size, 1])
