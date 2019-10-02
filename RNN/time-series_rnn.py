import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
tf.reset_default_graph()

df = pd.read_csv('petr4.csv')
df = df.dropna()

df = df.iloc[: , 1].values

periodos = 30
previsao_futura = 1 # horizonte das previsoes

X = df[0:(len(df) - (len(df) % periodos))]

X_batches = X.reshape(-1, periodos, 1)

y = df[1:(len(df) - (len(df) % periodos)) + previsao_futura]

y_batches = y.reshape(-1, periodos, 1)

X_teste = df[-(periodos + previsao_futura) : ]
X_teste = X_teste[:periodos]
X_teste = X_teste.reshape(-1, periodos, 1)

y_teste = df[-(periodos) : ]
y_teste = y_teste.reshape(-1, periodos, 1)

# -------------------------------------------
entradas = 1
neuronios_oculta = 100
neuronios_saida = 1

xph = tf.placeholder(tf.float32, [None, periodos, entradas])
yph = tf.placeholder(tf.float32, [None, periodos, neuronios_saida])

# Para criar uma celula apenas

# celula = tf.contrib.rnn.BasicRNNCell(num_units = neuronios_oculta, activation = tf.nn.relu)
# celula = tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)

# camada saida
# celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)

def cria_celula():
    return tf.contrib.rnn.BasicRNNCell(num_units = neuronios_oculta, activation = tf.nn.relu)
    # return tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)

def cria_varias_celulas(): # Cria 4 celulas
    return tf.nn.rnn_cell.MultiRNNCell([cria_celula() for i in range(4)])
    # celulas = tf.nn.rnn_cell.MultiRNNCell([cria_celula() for i in range(4)])
    # return tf.contrib.rnn.DropoutWrapper(celulas, output_keep_prob = 0.1)

celula = cria_varias_celulas()
# celula = tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)

# camada saida
celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)



saida_rnn, _ = tf.nn.dynamic_rnn(celula, xph, dtype = tf.float32)
erro = tf.losses.mean_squared_error(labels = yph, predictions = saida_rnn)
otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)

treinamento = otimizador.minimize(erro)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoca in range(1700):  # 3000 para RNNBasic e 1700 para LSTM
        _, custo = sess.run([treinamento, erro], feed_dict = {xph: X_batches, yph: y_batches})

        if epoca % 100 == 0:
            print("Epoca: " + str(epoca + 1) + " | Erro: " + str(custo))

    previsoes = sess.run(saida_rnn, feed_dict = {xph: X_teste})


y_teste2 = np.ravel(y_teste)
previsoes2 = np.ravel(previsoes)

mae = mean_absolute_error(y_teste2, previsoes2)
print("\nErro de: R${:.2f}").format(mae)

plt.plot(y_teste2, label = 'Valor real')
plt.plot(y_teste2, 'wo', markersize = 5, color = 'red')
plt.plot(previsoes2, label = 'Previsoes')
plt.legend()
plt.show()