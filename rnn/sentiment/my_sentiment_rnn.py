import tensorflow as tf
from constants import *
import prepare_data


def get_batches(arr, batch_size, sent_size):
    '''Создаем генератор, который возвращает пакеты размером
    n_seqs x n_steps из массива arr.
    
    Аргументы
    ---------
    arr: Массив, из которого получаем пакеты
    batch size: количество последовательностей в пакете
    sent_size: сколько "шагов" делаем в пакете
    '''
    # Считаем количество символов на пакет
    # и количество пакетов, которое можем сформировать
    characters_per_batch = n_seqs * n_steps
    n_batches = len(arr)//characters_per_batch
    
    # Сохраняем в массиве только символы, которые позволяют
    # сформировать целое число пакетов
    arr = arr[:n_batches * characters_per_batch]
    
    # Делаем reshape 1D -> 2D, используя n_seqs как число строк
    arr = arr.reshape((n_seqs, -1))
    
    for n in range(0, arr.shape[1], n_steps):
        # пакет данных, который будет подаваться на вход сети
        x = arr[:, n:n+n_steps]
        # целевой пакет, с которым будем сравнивать
        # предсказание, получаем сдвиганием "x" на один
        # символ вперед
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


def build_inputs(batch_size, sent_size):
    '''
    placeholders for inputs, targets and drop_out
    '''
    inputs = tf.placeholder(tf.int32, [batch_size,
                                       sent_size]
                            name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, 2],
                             name='targets')
    dropout_keep_prob = tf.placeholder(
        tf.float32, name='keep_prob')
    return inputs, targets, dropout_keep_prob


def build_lstm(lstm_size, num_layers, batch_size,
               dropout_keep_prob):
    def build_cell(lstm_size, dropout_keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(
            lstm, output_keep_prob=dropout_keep_prob)
        return drop
    
    cell = tf.contrib.rnn.MultiRNNCell(
        [build_cell(lstm_size, dropout_keep_prob) for _ in range(
            num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state


class LSTMNetwork (object):
    def __init__ (self, lstm_size, num_layers, embedding_size,
                  vocabulary_size, sent_size,
                  batch_size,
                  learning_rate =0.01):
        self.inputs, self.target, self.dropout_keep_prob = build_inputs(batch_size, sent_size)
        cell, self.initial_state = build_lstm(lstm_size,
                                              num_layers,
                                              batch_size,
                                              self.keep_prob)
        
        outputs, state = tf.nn.dynamic_rnn(
            cell, self.inputs, initial_state=self.initial_state)
        self.final_state = state
        self.predict, self.logits = build_output(
            outputs, lstm_size, num_classes)
        self.loss = build_loss(self.logits, self.target,
                               hidden_size, num_classes)
        tf.summary.scalar ('loss', self.losses)
        self.optimizer = build_optimizer(
            self.loss, learning_rate, grad_clip)
        correct_pred = tf.equal(tf.argmax (self.predict, 1),
                                tf.argmax (self.target, 1))
        self.accuracy = tf.reduce_mean (
            tf.cast(correct_pred, tf.float32),
            name='accuracy')
        tf.summary.scalar ('accuracy', self.accuracy)


class LSTMNetwork (object):
    def __init__ (self, lstm_size, num_layers, embedding_size,
                  vocabulary_size, sent_size,
                  batch_size,
                  learning_rate =0.01):
        # Параметры модели:
        # lstm_size: ко-о LSTM-модулей в блоке
        # num_layers: кол-во блоков
        # embedding_size: размер векторного представления слов
        # vocabulary_size: количество слов в словаре
        # sent_size: максимальная длина входного тензора
        # learning_rate: темп обучения метода оптимизации Adam
        # batch_size: кол-во статей за один полный проход сети
        
        self.inputs, self.target, self.dropout_keep_prob = build_inputs(batch_size, sent_size)
        
        # Слой векторного представления слов [vocabulary_size,
        # embedding_size]
        # to write
        self.word_embeddings = prepare_data.make_dictionary()
        
        # Создание LSTM-слоёв
        cell, self.initial_state = build_lstm(lstm_size,
                                              num_layers,
                                              batch_size,
                                              self.keep_prob)
        outputs, state = tf.nn.dynamic_rnn(
            cell, self.inputs, initial_state=self.initial_state)
        self.final_state = state
        
        # Построение полносвязного слоя
        self.predict, self.logits = build_output(
            outputs, lstm_size, num_classes)
        
        
        # Минимизация потерь перекрестной энтропии
        self.loss = build_loss(self.logits, self.target,
                               hidden_size, num_classes)
            tf.summary.scalar ('loss', self.losses)

        # Минимизация функции градиентного спуска
        self.optimizer = build_optimizer(
            self.loss, learning_rate, grad_clip)

        # Среднее значение точности для используемого батча
        correct_pred = tf.equal(tf.argmax (self.predict, 1),
                                tf.argmax (self.target, 1))
        self.accuracy = tf.reduce_mean (
            tf.cast(correct_pred, tf.float32),
            name='accuracy')
        tf.summary.scalar ('accuracy', self.accuracy)


def main():
    counter = 0
    rnn = LSTMNetwork(hidden_size=[hidden_size],
                      embedding_size=embedding_size,
                      vocabulary_size=get_data.vocab_size,
                      sent_size = max_length,
                      learning_rate =learning_rate)
            
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # x_validation, y_validation, validation_seq_len = get_data.get_validation_data()
    # train_writer = tf.summary.FileWriter('logs/train')
    # validation_writer = tf.summary.FileWriter('logs/validation')
    # train_writer.add_graph(rnn.input.graph)


    for i in range(epochs):
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run(
                [model.loss,
                 model.final_state,
                 model.optimizer],
                feed_dict=feed)
            print('Epoch: {}/{}... '.format(e+1, epochs),
                  'Training Step: {}... '.format(counter),
                  'Training loss: {:.4f}... '.format(batch_loss))
            if (counter % save_every_n == 0):
                check_p = "checkpoints/i{}_l{}.ckpt".format(
                    counter, lstm_size)
                print(check_p)
            
