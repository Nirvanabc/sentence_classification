from keras.layers import *
import prepare_data
import numpy as np
from constants import *
from keras.models import Model
from keras.utils import plot_model
from keras import losses, optimizers

keep_prob = 0.8

x_input = Input(shape=(sent_size, vec_size))
x = Reshape((sent_size, vec_size, 1))(x_input)

pooled_outputs = []

for i, filter_size in enumerate(filter_sizes):
    conv_i = Conv2D(
        filters=num_filters,
        kernel_size=(filter_size, vec_size),
        strides=(1, 1),
        padding='valid',
        activation='sigmoid',
        data_format="channels_last")(x)
    pooled_i = MaxPooling2D(
        pool_size=(sent_size - filter_size + 1, 1),
        padding='valid')(conv_i)
    pooled_outputs.append(pooled_i)

num_filters_total = num_filters * len(filter_sizes)
h_pool_k = Concatenate(axis=3)(pooled_outputs)
h_pool_flat_k = Reshape((num_filters_total))(h_pool_k)
h_dp = Dropout(1-keep_prob)(h_pool_flat_k)

output_layer = Dense(class_num)(h_dp)

model = Model(inputs=x_input, outputs=output_layer)

model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# def generate_arrays_from_file(path):
#     while True:
#         with open(path) as f:
#             for line in f:
#                 # create numpy arrays of input data
#                 # and labels, from each line in the file
#                 x1, x2, y = process_line(line)
#                 yield ({'input_1': x1, 'input_2': x2},
#                        {'output': y})
# 

model.fit_generator(
    prepare_data.generate_arrays_from_file(train_file),
    steps_per_epoch=1500, epochs=10, verbose=2)


plot_model(model, to_file='model.png', show_shapes=True)
