seq_len = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

batch_size = 100        # Размер пакета
num_steps = 100         # Шагов в пакете
lstm_size = 700 # 512  # Количество LSTM юнитов в скрытом слое
num_layers = 2          # Количество LSTM слоев
learning_rate = 0.001   # Скорость обучения
keep_prob = 0.5         # Dropout keep probability

epochs = 20
# Сохраняться каждый N итераций
save_every_n = 100
# количество сохранённых версий
max_to_keep=100

f = open('1_input.txt', 'r')
