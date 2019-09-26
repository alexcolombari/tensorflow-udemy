# Classificacao usando a base de dados de digitos mnist

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Download dataset do mnist digits
mnist = input_data.read_data_sets('mnist/', one_hot = True)

X_train = mnist.train.images
X_test = mnist.test.images
Y_train = mnist.train.labels
Y_test = mnist.test.labels

'''
amostra = 101
plt.imshow(X_train[amostra].reshape((28, 28)), cmap = 'gray')
plt.title('Classe: ' + str(np.argmax(Y_train[amostra])))
plt.show()
'''

# ---------------------------------------------
# NEURAL NET

# Estrutura: 784 -> 397 -> 397 -> 397 -> 10
# Neuronios de entrada
neuronios_entrada = X_train.shape[1]

# Camadas ocultas
neuronios_oculta1 = 397 # int((X_train.shape[1] + Y_train[1]) / 2)
neuronios_oculta2 = neuronios_oculta1
neuronios_oculta3 = neuronios_oculta1

# Camada de saida
neuronios_saida = Y_train.shape[1]

# Matriz de pesos (camada1 * camada2 -> camada2 * camada3 -> ...)
W = {'oculta1': tf.Variable(tf.random.normal([neuronios_entrada, neuronios_oculta1])),
     'oculta2': tf.Variable(tf.random.normal([neuronios_oculta1, neuronios_oculta2])),
     'oculta3': tf.Variable(tf.random.normal([neuronios_oculta2, neuronios_oculta3])),
     'saida': tf.Variable(tf.random.normal([neuronios_oculta3, neuronios_saida]))
     }

# Bias
b = {'oculta1': tf.Variable(tf.random.normal([neuronios_oculta1])),
     'oculta2': tf.Variable(tf.random.normal([neuronios_oculta2])),
     'oculta3': tf.Variable(tf.random.normal([neuronios_oculta3])),
     'saida': tf.Variable(tf.random.normal([neuronios_saida]))}

# Placeholders
xph = tf.compat.v1.placeholder('float', [None, neuronios_entrada])
yph = tf.compat.v1.placeholder('float', [None, neuronios_saida])


# Feed forward
# f(x) = (x * weight) + bias
# Relu activation
def mlp(x, W, bias):
    camada_oculta1 = tf.nn.relu(tf.add(tf.matmul(x, W['oculta1']), bias['oculta1']))
    camada_oculta2 = tf.nn.relu(tf.add(tf.matmul(camada_oculta1, W['oculta2']), bias['oculta2']))
    camada_oculta3 = tf.nn.relu(tf.add(tf.matmul(camada_oculta2, W['oculta3']), bias['oculta3']))
    camada_saida = tf.add(tf.matmul(camada_oculta3, W['saida']), bias['saida'])

    return camada_saida

modelo = mlp(xph, W, b)

# Calculo do erro (usando Softmax cross entropy)
erro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = modelo, labels = yph))
# otimizador usando Algoritmo de Adam e minimizando o erro
otimizador = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.0001).minimize(erro)

# Predict
# softmax utilizado para previsao das classes
previsoes = tf.nn.softmax(modelo)

# Comparando as previsoes com a resposta real
previsoes_corretas = tf.equal(tf.argmax(previsoes, 1), tf.argmax(yph, 1))

# taxa de acerto
taxa_acerto = tf.reduce_mean(tf.cast(previsoes_corretas, tf.float32))

# Tensorflow session for training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('\n\nInicio do treinamento\n')
    for epoca in range(15000):
        # BATCH SIZE
        X_batch, y_batch = mnist.train.next_batch(128)
        _, custo = sess.run([otimizador, erro], feed_dict = {xph: X_batch, yph: y_batch})
        if epoca % 100 == 0:
            # calculo acuracia
            acc = sess.run([taxa_acerto], feed_dict = {xph: X_batch, yph: y_batch})
            print('epoca: ' + str(epoca) + '  - erro: ' + str(custo) + '  - acc: ' + str(acc))
    
    print('\nTreinamento concluido\n')
    print("Acuracia na base de dados de teste: " + str(sess.run(taxa_acerto, feed_dict = {xph: X_test, yph: Y_test})))
