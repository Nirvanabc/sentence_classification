import tensorflow as tf
from constants import *
from prepare_data import *


def build_inputs(batch_size, sent_size,
                 embedding_size, class_num):
    '''
    placeholders for inputs, targets and dropout
    '''
    inputs = tf.placeholder(tf.int32, [batch_size,
                                       sent_size,
                                       embedding_size]
                            name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, class_num],
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
    softmax_w = tf.Variable(tf.truncated_normal(
        (in_size, out_size), stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.matmul(x, softmax_w) + softmax_b
    ## FIXME! выяснить, зачем softmax, если в build_loss
    # есть softmax_cross_entropy_with_logits
    
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits


class LSTMNetwork (object):
    def __init__ (self, lstm_size, num_layers, embedding_size,
                  sent_size, batch_size, class_num,
                  learning_rate =0.01):
        # Параметры модели:
        # lstm_size: ко-о LSTM-модулей в блоке
        # num_layers: кол-во блоков
        # embedding_size: размер векторного представления слов
        # batch_size: кол-во предлож-й за один полный проход сети
        # class_num количество классов классификации
        # sent_size: максимальная длина входного тензора
        # learning_rate: темп обучения метода оптимизации Adam

        # предложения поступают уже в вектроном виде
        
        self.inputs, self.target, self.dropout_keep_prob = build_inputs(batch_size, sent_size, embedding_size, class_num)
        
        # Создание LSTM-слоёв
        cell, self.initial_state = build_lstm(
            lstm_size,
            num_layers,
            batch_size,
            self.dropout_keep_prob)
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
            
