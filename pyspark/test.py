from bigdl.nn.layer import *
import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

SUBMIT_ARGS = "" \
              "--conf spark.driver.extraClassPath=/home/yang/sources/bigdl/dist/lib/bigdl-0.3.0-SNAPSHOT-jar-with-dependencies.jar" \
              " pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS







def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_b):
    h = tf.nn.bias_add(tf.matmul(X, w_h), w_b) # this is a basic mlp, think 2 stacked logistic regressions
    return h # note that we dont take the softmax at the end because our cost fn does that for us


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 10]) # create symbolic variables
w_b = init_weights([10])

py_x = model(X, w_h, w_b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)




model = Model(X, py_x, "float", "little_endian", "tensorflow")



# layer = Linear(1, 5)

print model