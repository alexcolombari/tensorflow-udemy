import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------- Load data --------------------
iris = datasets.load_iris()
x = iris.data
y = iris.target

# -------------------- X Scaler --------------------
scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

# -------------------- One Hoting --------------------
onehot = OneHotEncoder(categorical_features = [0])
y = y.reshape(-1, 1)
y = onehot.fit_transform(y).toarray()

# -------------------- Spliting --------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# -------------------- Neural Net --------------------
input_neurons = x.shape[1]
hidden_neurons = int(np.ceil((x.shape[1] + y.shape[1]) / 2))
output_neurons = y.shape[1]

# Weights random values
weights = {'hidden:': tf.Variable(tf.random_normal([input_neurons, hidden_neurons])),
            'output': tf.Variable(tf.random_normal([hidden_neurons, output_neurons]))
}

# Bias random values
bias = {'hidden': tf.Variable(tf.random_normal([hidden_neurons])),
        'output': tf.Variable(tf.random_normal([output_neurons]))
}

# Placeholders
xph = tf.placeholder('float', [None, input_neurons])
yph = tf.placeholder('float', [None, output_neurons])

# Model
def mlp(x, weights, bias):
    hidden_neurons = tf.add(tf.matmul(x, weights['hidden']), bias['hidden'])
    hidden_neurons_activation = tf.nn.relu(hidden_neurons)
    output_neurons = tf.add(tf.matmul(hidden_neurons_activation, weights['output']), bias['output'])
    return output_neurons

model = mlp(xph, weights, bias)
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = model, labels = yph))
optmizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(error)

batch_size = 8
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(3000):
        mean_error = 0
        total_batch = int(len(x_train) / batch_size)
        x_batches = np.array_split(x_train, total_batch)
        y_batches = np.array_split(y_train, total_batch)
        for i in range(total_batch):
            x_batch, y_batch = x_batches[i], y_batches[i]
            _, cost = sess.run([optmizer, error], feed_dict = {xph: x_batch, yph: y_batch})
            mean_error += cost / total_batch
            if epoch % 500 == 0:
                print('Epoch: ' + str(epoch + 1) + 'Error: ' + str(mean_error))
        