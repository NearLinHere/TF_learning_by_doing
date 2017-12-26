# this sample is from tensorflow
# https://www.tensorflow.org/get_started/get_started
# to understand variables, loss function, and optimizer
# For complete version, please refer easy_simple_regression_2.py

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)  # input data
y = tf.placeholder(tf.float32)  # output data should be

linar_model = W*x + b

# count square error for each output value and y
squared_deltas = tf.square(linar_model - y )

# reduce_sum: only sum all the squared error
# training part is charged by the optimizer
loss = tf.reduce_sum(squared_deltas)

# the optimizer is heading for minimize loss
# tuning the tf.Varialbe
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Count loss
# print(sess.run(loss, {x: [1, 2, 3, 4], y:[0, -1, -2, -3]}))

for i in range(1000):
    sess.run(train,  {x: [1, 2, 3, 4], y:[0, -1, -2, -3]})

# y = -x + 1
print (sess.run([W, b]))