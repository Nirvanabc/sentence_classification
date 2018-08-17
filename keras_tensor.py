# the same as tensor.py but using keras

from keras.layers import *
from prepare_data import *
import numpy as np
from constants import *
from keras.models import Model
from keras.utils import plot_model
from keras import losses, optimizers
import tensorflow as tf

keep_prob = 0.5

x_input = Input(shape=(sent_size, vec_size))
x = Reshape((sent_size, vec_size, 1))(x_input)

pooled_outputs = []

for i, filter_size in enumerate(filter_sizes):
    conv_i = Conv2D(
        filters=num_filters,
        kernel_size=(filter_size, vec_size),
        strides=(1, 1),
        padding='valid',
        data_format="channels_last")(x)
    activ = Activation('sigmoid')(conv_i)
    pooled_i = MaxPooling2D(
        pool_size=(sent_size - filter_size + 1, 1),
        padding='valid')(conv_i)
    pooled_outputs.append(pooled_i)

num_filters_total = num_filters * len(filter_sizes)
h_pool_k = Concatenate(axis=3)(pooled_outputs)
h_pool_flat_k = Reshape((num_filters_total, ))(h_pool_k)
h_dp = Dropout(1-keep_prob)(h_pool_flat_k)

output_layer = Dense(class_num)(h_dp)

model = Model(inputs=x_input, outputs=output_layer)


def mean_pred(y_true, y_pred):
    correct_prediction = tf.equal(tf.argmax(y_true, 1), \
                                 tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, \
                                      tf.float32))
    return accuracy


model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

new_batch_gen = next_batch(test_file, test_batch_size, vec_size)
new_batch = next(new_batch_gen)

# you can see this object keys by this:
# print(history.history.keys())

history = model.fit_generator(
    generator = generate_arrays_from_file(train_file, batch_size),
    validation_data = ([new_batch[0]], [new_batch[1]]),
    steps_per_epoch = 150,
    epochs = 30,
    verbose = 2)

# you can visualize the model arch
# plot_model(model, to_file='model.png', show_shapes=True)
