import tensorflow as tf


class LSTMNetwork (object):
    def __init__ (self, hidden_size, embedding_size,
                  vocabulary_size, max_length,
                  learning_rate =0.01) :
        # Параметры модели:
        # hidden_size: количество LSTM-модулей в блоке
        # embedding_size: размер векторного представления слов
        # vocabulary_size: количество слов в словаре
        # max_length: максимальная длина входного тензора
        # learning_rate: темп обучения метода оптимизации Adam
        # Плэйсхолдер входных данных [batch_size, max_length]
        self.input = tf.placeholder (tf.int32,
                                       [None, max_length],
                                       name='input')

        # Плэйсхолдер длины последовательности, имеющий
        # размерность [batch_size].Содержит длину каждого
        # тензора батча
        self.seq_len = tf.placeholder (tf.int32, [None],
                                         name='lengths')
        # Плэйсхолдер целевых значений [batch_size, 2]
        self.target = tf.placeholder (tf.float32, [None, 2],
                                        name ='target')
        # коэффициент дропаута
        self.dropout_keep_prob = tf.placeholder (
            tf.float32, name ='dropout_keep_prob')

        
        # Слой векторного представления слов [vocabulary_size,
        # embedding_size]
        with tf.name_scope ('word_embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform(
                    [vocabulary_size,
                     embedding_size ], -1, 1, seed=seed))
            self.word_embeddings = embedded_words = tf.nn.
            embedding_lookup(embeddings, x)

        # Создание LSTM-слоёв
        outputs = embedded_words
        for h in hidden_size:
            outputs = self._rnn_layer (h, outputs, seq_len,
                                        dropout_keep_prob)
        outputs = tf.reduce_mean (outputs,
                                  reduction_indices =[1])
        # Построение полносвязного слоя
        with tf.name_scope ('final_layer / weights'):
            w = tf.Variable (tf.truncated_normal(
                [self.hidden_size
                 [ − 1], 2]))
            self.variable_summaries (w,'final_layer / weights')
            
        with tf.name_scope ('final_layer / biases'):
            b = tf.Variable (tf.constant (0.1, shape =[2]))
            self.variable_summaries (b,'final_layer / biases')
            # Линейные активации для каждого класса
            # [batch_size, 2]

        with tf.name_scope ('final_layer / wx_plus_b'):
            self.activations = tf.nn.xw_plus_b (
                outputs, w, b, name='activations')
        
        # Софтмакс активации для каждого класса [batch_size, 2]
        self.predict = tf.nn.softmax (self.activations,
                                        name='predictions')
        
        # Минимизация потерь перекрестной энтропии [batch_size]
        self.losses = tf.nn.softmax_cross_entropy_with_logits (
            logits =self.activations, labels =self.target,
            name='cross_entropy')
        
        # Среднее значение потерь перекрестной энтропии
        with tf.name_scope ('loss'):
            self.loss = tf.reduce_mean (self.losses, name='loss')
            tf.summary.scalar ('loss', self.loss)
            # Минимизация функции градиентного спуска
            self.train_step = tf.train.AdamOptimizer (learning_rate).
            minimize (self.loss)
            # Среднее значение точности для используемого батча
            
        with tf.name_scope ('accuracy'):
            correct_pred = tf.equal(tf.argmax (self.predict, 1),
                                    tf.argmax (self.target, 1))
            self.accuracy = tf.reduce_mean (tf.cast(correct_pred,
                                                       tf.float32), name='accuracy')
            tf.summary.scalar ('accuracy', self.accuracy)

        def _cell(self, hidden_size, dropout_keep_prob, seed=None):
            # Построение LSTM-блока
            lstm_cell = tf.contrib.rnn.LSTMCell (hidden_size,
                                                     state_is_tuple =True)
            dropout_cell = tf.contrib.rnn.DropoutWrapper (
                lstm_cell,
                input_keep_prob = dropout_keep_prob, output_keep_prob =
                dropout_keep_prob, seed=seed)
        return dropout_cell

        def _rnn_layer (self, hidden_size, x, seq_len,
                        dropout_keep_prob, variable_scope =None):
            # Построение RNN слоя с LSTM-блоками
            with tf.variable_scope (variable_scope,
                                    default_name ='rnn_layer'):
                lstm_cell = self._cell(hidden_size, dropout_keep_prob)
                outputs, _ = tf.nn.dynamic_rnn (
                    lstm_cell, x, dtype=tf.float32, sequence_length = seq_len)
            return outputs
        
