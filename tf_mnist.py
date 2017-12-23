from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from conv_net import deepnn

learning_rate = 0.5
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)


def model(x, y, y_pared):
    soft_max = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pared)
    cross_entropy = tf.reduce_mean(soft_max)

    # Define a gradient step operation.
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pared, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Define a TensorFlow session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for _ in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(200)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        if _ % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
            validation_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
            print('step %d, training accuracy %g, validation accuracy %g' % (_, train_accuracy, validation_accuracy))

    # Test trained model
    print('Test Accuracy')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))


def assignment_3_a():
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.random_normal(([784, 10]), stddev=0.1))
    b = tf.Variable(tf.random_normal(([10]), stddev=0.1))

    # Output tensor.
    y_pared = tf.matmul(x, w) + b

    # Define loss and optimizer
    y = tf.placeholder(tf.float32, [None, 10])
    model(x, y, y_pared)


def assignment_3_b():
    x = tf.placeholder(tf.float32, [None, 784])
    n_hidden_size = 100
    w1 = tf.Variable(tf.random_normal(([784, n_hidden_size])))
    w2 = tf.Variable(tf.random_normal(([n_hidden_size, 10])))
    b1 = tf.Variable(tf.random_normal(([n_hidden_size])))
    b2 = tf.Variable(tf.random_normal(([10])))

    # action factions.
    z1 = tf.matmul(x, w1) + b1
    a1 = tf.nn.relu(z1)

    # Output tensor.
    y_pared = tf.matmul(a1, w2) + b2

    # Define loss and optimizer
    y = tf.placeholder(tf.float32, [None, 10])
    model(x, y, y_pared)


def assignment_3_c():
    x = tf.placeholder(tf.float32, [None, 784])
    n_hidden_size = 100
    w1 = tf.get_variable("w1", shape=[784, n_hidden_size],
                         initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable("w2", shape=[n_hidden_size, 10],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", shape=[n_hidden_size],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", shape=[10],
                         initializer=tf.contrib.layers.xavier_initializer())

    # action factions.
    z1 = tf.matmul(x, w1) + b1
    a1 = tf.nn.relu(z1)

    # Output tensor.
    y_pared = tf.matmul(a1, w2) + b2

    # Define loss and optimizer
    y = tf.placeholder(tf.float32, [None, 10])

    model(x, y, y_pared)


def assignment_3_d():
    x = tf.placeholder(tf.float32, [None, 784])

    # Output tensor.
    y_pared = deepnn(x)

    # Define loss and optimizer
    y = tf.placeholder(tf.float32, [None, 10])

    model(x, y, y_pared)


assignment_3_d()
