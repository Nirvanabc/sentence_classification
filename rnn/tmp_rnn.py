import tensorflow as tf
import numpy as np
from constants import *
import time


# def weight_variable(shape, name):
#     initial = tf.truncated_normal(shape, stddev=0.1, name=name+'_const')
#     return tf.Variable(initial, name=name)
# 
# def bias_variable(shape, name):
#     initial = tf.constant(0.1, shape=shape, name=name+'_const')
#     return tf.Variable(initial, name=name)


text=f.read()
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)


def get_batches(arr, n_seqs, n_steps):
    '''Создаем генератор, который возвращает пакеты размером
    n_seqs x n_steps из массива arr.
    
    Аргументы
    ---------
    arr: Массив, из которого получаем пакеты
    n_seqs: Batch size, количество последовательностей в пакете
    n_steps: Sequence length, сколько "шагов" делаем в пакете
    '''
    # Считаем количество символов на пакет и количество пакетов, которое можем сформировать
    characters_per_batch = n_seqs * n_steps
    n_batches = len(arr)//characters_per_batch
    
    # Сохраняем в массиве только символы, которые позволяют сформировать целое число пакетов
    arr = arr[:n_batches * characters_per_batch]
    
    # Делаем reshape 1D -> 2D, используя n_seqs как число строк
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        # пакет данных, который будет подаваться на вход сети
        x = arr[:, n:n+n_steps]
        # целевой пакет, с которым будем сравнивать предсказание, получаем сдвиганием "x" на один символ вперед
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y

        
def build_inputs(batch_size, num_steps):
    ''' Определяем placeholder'ы для входных, целевых данных, а также вероятности drop out

    Аргументы
    ---------
    batch_size: Batch size, количество последовательностей в пакете
    num_steps: Sequence length, сколько "шагов" делаем в пакете
    
    '''
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    # Placeholder для вероятности drop out
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob

        
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
    # with tf.variable_scope('softmax'):
    softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.matmul(x, softmax_w) + softmax_b
    ## FIXME! выяснить, зачем softmax, если в build_loss
    # есть softmax_cross_entropy_with_logits
    
    out = tf.nn.softmax(logits, name='predictions')
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

class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2,
                 learning_rate=0.001,
                 grad_clip=5, sampling=False):
        
        # Мы будем использовать эту же сеть для сэмплирования
        # (генерации текста),
        # при этом будем подавать по одному символу за один раз
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
            
        tf.reset_default_graph()
        
        # Получаем input placeholder'ы
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        
        # Строим LSTM ячейку
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        
        ### Прогоняем данные через RNN слои
        # Делаем one-hot кодирование входящих данных
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # Прогоняем данные через RNN и собираем результаты
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        # Получаем предсказания (softmax) и результат logit-функции
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        
        # Считаем потери и оптимизируем (с обрезкой градиента)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

            

# saver.restore(sess, 'checkpoints/______.ckpt')

def main(check_p = 0):
    counter = 0
    
    model = CharRNN(len(vocab), batch_size=batch_size,
                    num_steps=num_steps,
                    lstm_size=lstm_size, num_layers = num_layers,
                    learning_rate=learning_rate)
    
    saver = tf.train.Saver(max_to_keep=100)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Можно раскомментировать строчку ниже и продолжить
    # обучение с checkpoint'а
    saver.restore(sess, check_p)
    
    test_writer = tf.summary.FileWriter(
        "output/", sess.graph)
    
    tf.summary.scalar('batch_loss', model.loss)
    merged = tf.summary.merge_all()
    
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
            batch_loss, new_state, summary, _ = sess.run(
                [model.loss,
                 model.final_state,
                 merged,
                 model.optimizer],
                feed_dict=feed)
    
            test_writer.add_summary(summary, counter)
            end = time.time()
            
            print('Epoch: {}/{}... '.format(e+1, epochs),
                  'Training Step: {}... '.format(counter),
                  'Training loss: {:.4f}... '.format(batch_loss),
                  '{:.4f} sec/batch'.format((end-start)))
            
            if (counter % save_every_n == 0):
                check_p = "checkpoints/i{}_l{}.ckpt".format(
                    counter, lstm_size)
                print(check_p)
                saver.save(sess, check_p)
                # # to see the result every n steps uncomment:
                # sess.close()
                # print_sample(check_p)
                # model = CharRNN(len(vocab),
                #     batch_size=batch_size,
                #     num_steps=num_steps,
                #     lstm_size=lstm_size,
                #     num_layers = num_layers,
                #     learning_rate=learning_rate)
                # 
                # # saver = tf.train.Saver(max_to_keep=100)
                # # tf.summary.scalar('batch_loss', model.loss)
                # # merged = tf.summary.merge_all()
                # init = tf.global_variables_initializer()
                # sess = tf.Session()
                # sess.run(init)
                # saver.restore(sess, check_p)
                
    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(
        counter, lstm_size))
