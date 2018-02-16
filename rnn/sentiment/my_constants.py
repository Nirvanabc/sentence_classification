label_bad = 0
label_good = 1

embedding_size = 100

# кол-во слов в самом длинном предложении
max_length = 30 # number of steps to unroll the RNN for

seq_len = 50 # batch_size
num_classes = 2

keep_prob = 0.5
# hidden_size: количество LSTM-модулей в блоке


learning_rate = 1e-1

lstm_size = 500 # 512  # Количество LSTM юнитов в скрытом слое
num_layers = 2          # Количество LSTM слоев
learning_rate = 0.001   # Скорость обучения
keep_prob = 0.5         # Dropout keep probability

epochs = 20
# Сохраняться каждый N итераций
save_every_n = 100
# количество сохранённых версий
max_to_keep=100
