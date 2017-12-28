from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from conv_net import deepnn

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
dir_path = repr(os.path.dirname(os.path.realpath(sys.argv[0]))).strip("'")


def model(x, y, y_pared, learning_rate=0.5, optimizer=tf.train.GradientDescentOptimizer, batch_size=200,
          steps_number=3000):
    soft_max = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pared)
    cross_entropy = tf.reduce_mean(soft_max)

    # Define a gradient step operation.
    train_step = optimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pared, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Define a TensorFlow session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for _ in range(steps_number):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        if _ % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
            validation_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
            print('step %d, training accuracy %g, validation accuracy %g' % (_, train_accuracy, validation_accuracy))

    # Test trained model
    print('Test Accuracy')
    current_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print(current_accuracy)
    sess.close()
    tf.reset_default_graph()
    return current_accuracy


def assignment_3_a():
    print('Testing with random initializer with std=0.1 and mean=0: ')

    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.random_normal(([784, 10]), stddev=0.1))
    b = tf.Variable(tf.random_normal(([10]), stddev=0.1))

    # Output tensor.
    y_pared = tf.matmul(x, w) + b

    # Define loss and optimizer
    y = tf.placeholder(tf.float32, [None, 10])
    model(x, y, y_pared)


def assignment_3_b():
    print('Testing with zero initializer: ')

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
    print('Testing xavier_initializer: ')

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
    print('Testing conv-net: ')

    x = tf.placeholder(tf.float32, [None, 784])

    y_pared = deepnn(x)

    y = tf.placeholder(tf.float32, [None, 10])

    model(x, y, y_pared, learning_rate=0.0001, optimizer=tf.train.AdamOptimizer)


def assignment_3_e():
    learning_rate_options = [np.math.pow(10, i) for i in np.linspace(-5, -3, 10)]
    number_of_repeats = 10
    validation_result_accuracy = []
    for learning_rate in learning_rate_options:
        accuracy = 0
        for _ in range(number_of_repeats):
            print('Testing learning rate: ' + str(learning_rate))
            x = tf.placeholder(tf.float32, [None, 784])

            y_pared = deepnn(x)

            y = tf.placeholder(tf.float32, [None, 10])

            accuracy += model(x, y, y_pared, learning_rate=0.0001, optimizer=tf.train.AdamOptimizer, batch_size=10,
                              steps_number=1000)

        validation_result_accuracy.append(float(accuracy) / float(number_of_repeats))
    max_value = max(validation_result_accuracy)
    max_index = validation_result_accuracy.index(max_value)
    my_best_learning_rate = learning_rate_options[max_index]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.plot(learning_rate_options, validation_result_accuracy, 'b-', label='Validation Accuracy')
    plt.legend()
    plt.xlabel('learning rate', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    fig.savefig(os.path.join(dir_path, '5_a.png'))
    fig.clf()
    print('done')
    print("Best learning rate was: " + str(my_best_learning_rate))

    print("Testing the best learning rete: ")
    x = tf.placeholder(tf.float32, [None, 784])
    y_pared = deepnn(x)
    y = tf.placeholder(tf.float32, [None, 10])
    model(x, y, y_pared, learning_rate=my_best_learning_rate, optimizer=tf.train.AdamOptimizer)


if len(sys.argv) < 2:
    print("Please enter which part do you want to execute - a, b, c, d, e or all")
    exit()

cmds = sys.argv[1:]
for cmd in cmds:
    if cmd not in ['a', 'b', 'c', 'd', 'e', 'all']:
        print("Unknown argument %s. please run with a, b, c, d,e or all" % cmd)
        exit()

if 'a' in cmds or 'all' in cmds:
    assignment_3_a()
if 'b' in cmds or 'all' in cmds:
    assignment_3_b()
if 'c' in cmds or 'all' in cmds:
    assignment_3_c()
if 'd' in cmds or 'all' in cmds:
    assignment_3_d()
if 'e' in cmds or 'all' in cmds:
    assignment_3_e()
