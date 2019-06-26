import numpy as np
import tensorflow as tf

x = np.array([[0.0, 0.0],
             [0.0, 1.0],
             [1.0, 0.0],
             [1.0, 1.0]])
        
y = np.array([[0.0], [0.0], [0.0], [1.0]])

W = tf.Variable(tf.zeros([2, 1], dtype = tf.float64))
init = tf.global_variables_initializer()

def step(x):
    return tf.cast(tf.to_float(tf.math.greater_equal(x, 1)), tf.float64)

output = tf.matmul(x, W)
output_activation = step(output)
error = tf.subtract(y, output_activation)

delta = tf.matmul(x, error, transpose_a = True)
learning_rate = tf.assign(W, tf.add(W, tf.multiply(delta, 0.1)))

with tf.Session() as sess:
    sess.run(init)
    #print('\n', sess.run(W))
    #print('\n', sess.run(output_activation))
    #print('\n', sess.run(error))
    #print(sess.run(learning_rate))
    epoch = 0
    for i in range(15):
        epoch += 1
        error_result, _ = sess.run([error, learning_rate])
        error_sum = tf.reduce_sum(error_result)
        print('Epoch: {}, Error: {}'.format(epoch, sess.run(error_sum)))
        if error_sum.eval() == 0:
            break
    final_weights = W
    print(sess.run(final_weights))


output_test = tf.matmul(x, final_weights)
output_activation_test = step(output_test)
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(output_activation_test))