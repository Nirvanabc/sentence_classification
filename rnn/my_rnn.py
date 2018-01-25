import tensorflow as tf
import numpy as np
from constants import *


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (
    data_size, vocab_size))

inputs = tf.placeholder(tf.int32, [batch_size, seq_len])
targets = tf.placeholder(tf.int32, [batch_size, seq_len])
keep_prob = tf.placeholder(tf.float32)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    def build_cell(lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(
            lstm, output_keep_prob=keep_prob)
        return drop
    
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(
        lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state


def build_output(lstm_output, in_size, out_size):
    ''' Строим softmax слой и возвращаем результат его работы.
        
    lstm_output: Входящий от LSTM тензор
    in_size: Размер входящего тензора, (кол-во LSTM юнитов 
    скрытого слоя)
    out_size: Размер softmax слоя (объем словаря)
    '''
    # вытягиваем и решэйпим тензор, выполняя  3D -> 2D
    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])
    softmax_w = weight_variable([in_size, out_size])
    softmax_b = bias_variable(out_size)
    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits)
    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    ''' Считаем функцию потери на основании значений 
    logit-функции и целевых значений.
    
    Аргументы
    ---------
    logits: значение logit-функции
    targets: целевые значения, с которыми сравниваем предсказания
    lstm_size: Количество юнитов в LSTM слое
    num_classes: Количество классов в целевых значениях 
    (размер словаря)
    '''
    # Делаем one-hot кодирование целевых значений и
    # решейпим по образу и подобию logits
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    
    # Считаем значение функции потери
    # softmax cross entropy loss и возвращаем среднее значение
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss


def build_optimizer(loss, learning_rate, grad_clip):
    ''' Строим оптимизатор для обучения, используя обрезку
    градиента.
    
    Arguments:
    loss: значение функции потери
    learning_rate: параметр скорости обучения
    '''
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    return optimizer


def charRNN(num_classes, batch_size=64, num_steps=50,
             lstm_size=128, num_layers=2, learning_rate=0.001,
             grad_clip=5, sampling=False):
    # Мы будем использовать эту же сеть для генерации текста,
    # при этом будем подавать по одному символу за один раз
    if sampling == True:
        batch_size, num_steps = 1, 1
    else:
        batch_size, num_steps = batch_size, num_steps
        tf.reset_default_graph()
        
        # Получаем input placeholder'ы
        inputs, targets, keep_prob = build_inputs(
            batch_size, num_steps)
        
        # Строим LSTM ячейку
        cell, initial_state = build_lstm(
            lstm_size, num_layers, batch_size, keep_prob)
        
        ## Прогоняем данные через RNN слои
        # Делаем one-hot кодирование входящих данных
        x_one_hot = tf.one_hot(inputs, num_classes)
        
        # Прогоняем данные через RNN и собираем результаты
        outputs, state = tf.nn.dynamic_rnn(
            cell, x_one_hot, initial_state = initial_state)
        final_state = state
        
        # Получаем предсказания (softmax) и рез-т logit-функции
        prediction, logits = build_output(
            outputs, lstm_size, num_classes)
        
        # Считаем потери и оптимизируем (с обрезкой градиента)
        loss = build_loss(
            logits, targets, lstm_size, num_classes)
        optimizer = build_optimizer(
            loss, learning_rate, grad_clip)
            
model = CharRNN(len(vocab), batch_size=batch_size,
                num_steps=num_steps,
                lstm_size=lstm_size, num_layers = num_layers,
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
        

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Можно раскомментировать строчку ниже и продолжить
    # обучение с checkpoint'а
    # saver.restore(sess, 'checkpoints/______.ckpt')
    counter = 0
    for e in range(epochs):
        # Обучаем сеть
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run(
                [model.loss,
                 model.final_state,
                 model.optimizer],
                feed_dict=feed)
            
            end = time.time()
            print('Epoch: {}/{}... '.format(e+1, epochs),
                  'Training Step: {}... '.format(counter),
                  'Training loss: {:.4f}... '.format(batch_loss),
                  '{:.4f} sec/batch'.format((end-start)))
            
            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
                
    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))


# def pick_top_n(preds, vocab_size, top_n=5):
#     '''
#     оставляет 5 наиболее вероятных букв.
#     '''
#     p = np.squeeze(preds)
#     p[np.argsort(p)[:-top_n]] = 0
#     p = p / np.sum(p)
#     c = np.random.choice(vocab_size, 1, p=p)[0]
#     return c
# 
# 
# def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="Гостиная Анны Павловны начала понемногу наполняться."):
#     samples = [c for c in prime]
#     model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, checkpoint)
#         new_state = sess.run(model.initial_state)
#         for c in prime:
#             x = np.zeros((1, 1))
#             x[0,0] = vocab_to_int[c]
#             feed = {model.inputs: x,
#                     model.keep_prob: 1.,
#                     model.initial_state: new_state}
#             preds, new_state = sess.run([model.prediction, model.final_state],
#                                         feed_dict=feed)
#             
#         c = pick_top_n(preds, len(vocab))
#         samples.append(int_to_vocab[c])
#         for i in range(n_samples):
#             x[0,0] = c
#             feed = {model.inputs: x,
#                     model.keep_prob: 1.,
#                     model.initial_state: new_state}
#             preds, new_state = sess.run([model.prediction, model.final_state],
#                                         feed_dict=feed)
#             
#             c = pick_top_n(preds, len(vocab))
#             samples.append(int_to_vocab[c])
#             
#     return ''.join(samples)
#     
# checkpoint = 'checkpoints/i200_l512.ckpt'
# samp = sample(checkpoint, 1000, lstm_size, len(vocab))
# print(samp)
