# see Fanifatieva for details
import tensorflow as tf
input_data = tf.placeholder(tf.int32, [None, max_length])

# batch_size
seq_len = tf.placeholder(tf.int32, [None])

# y_
target = tf. placeholder(tf.float32, [None , 2])

dropout_keep_prob = tf.placeholder(tf.float32)

# LSTM layers
for h in hidden_size:
    outputs = _rnn_layer(h, outputs, seq_len,
                                dropout_keep_prob)

outputs = tf.reduce_mean(outputs, reduction_indices =[1])

# fully connected layer
w = tf.Variable(tf.truncated_normal([hidden_size[-1], 2]))
# variable_summaries(w, 'final_layer/weights')

b = tf.Variable(tf.constant (0.1, shape =[2]))
# variable_summaries(b, 'final_layer/biases')

# linear activations
activations = tf.nn.xw_plus_b(outputs, w, b)
predict = tf.nn.softmax(activations)
losses = tf.nn. softmax_cross_entropy_with_logits(
    logits = activations, labels = target)

loss = tf.reduce_mean(losses , name='loss')
tf.summary.scalar('loss', loss)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax (predict, 1),
                        tf.argmax (target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,
                                  tf.float32))
tf.summary.scalar('accuracy', accuracy)

def cell(hidden_size, dropout_keep_prob, seed=None):
    lstm_cell = tf.contrib.rnn.LSTMCell(hidden_size,
                                        state_is_tuple = True)
    dropout_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_cell,
        input_keep_prob = dropout_keep_prob,
        output_keep_prob = dropout_keep_prob,
        seed=seed)
    return dropout_cell


def _rnn_layer (hidden_size, x, seq_len,
                 dropout_keep_prob, variable_scope = None):
    # Построение RNN слоя с LSTM-блоками
    lstm_cell = cell(hidden_size, dropout_keep_prob)
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x,
                                   dtype=tf.float32,
                                   sequence_length = seq_len)
    return outputs

rnn = LSTMNetwork(hidden_size = [FLAGS.hidden_size],
                  embedding_size = FLAGS.embedding_size,
                  vocabulary_size = get_data .vocab_size ,
                  max_length = get_data.sequence_len,
                  learning_rate =FLAGS.learning_rate )

sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
x_validation, y_validation, validation_seq_len = get_data.get_validation_data()
train_writer = tf.summary.FileWriter('logs/train')
validation_writer = tf.summary.FileWriter('logs/validation')
train_writer.add_graph(rnn.input.graph)

for i in range(FLAGS.train_steps):
    # Perform training step
    x_train, y_train, train_seq_len = get_data.batch(
        FLAGS.batch_size)
    train_loss , _, summary = sess.run(
        [rnn.loss, rnn.train_step,
         tf.summary.merge_all ()],
        feed_dict ={
            rnn.input: x_train,
            rnn.target: y_train,
            rnn.seq_len: train_seq_len,
            rnn.dropout_keep_prob: FLAGS.dropout_keep_prob })
    train_writer.add_summary(summary, i)
    if (i + 1) % FLAGS.validate_every == 0:
        validation_loss, accuracy, summary = sess.run(
            [rnn.loss, rnn.accuracy, tf.summary.merge_all()],
            feed_dict ={ rnn.input: x_validation,
                         rnn.target: y_validation,
                         rnn.seq_len: validation_seq_len,
                         rnn.dropout_keep_prob: 1})
        validation_writer.add_summary(summary, i)
