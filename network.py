"""
network.py
"""

import random
import numpy as np
import math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

dir_path = repr(os.path.dirname(os.path.realpath(sys.argv[0]))).strip("'")

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data, plot_graphs = False):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  """
        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))
        n = len(training_data)
        training_accuracy = []
        training_loss = []
        test_accuracy = []
        epochs = [i for i in range(epochs)]
        for epoch in epochs:
            random.shuffle(list(training_data))
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if(plot_graphs):
                training_accuracy.append(self.one_hot_accuracy(training_data))
                training_loss.append(self.loss(training_data))
                test_accuracy.append(self.one_label_accuracy(test_data))
            print ("Epoch {0} test accuracy: {1}".format(epoch, self.one_label_accuracy(test_data)))

        if (plot_graphs):
            self.plot_graph(epochs, training_accuracy, 'training accuracy', 'epcohs', 'training accuracy', '2_b_training_accuracy.png')
            self.plot_graph(epochs, training_loss, 'training loss', 'epcohs', 'training loss', '2_b_training_loss.png')
            self.plot_graph(epochs, test_accuracy, 'test accuracy', 'epcohs', 'test accuracy', '2_b_test_accuracy.png')

    def plot_graph(self, x, y, label, xlabel, ylabel,name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y, label=label)
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=16)
        fig.savefig(os.path.join(dir_path, name))
        fig.clf()

    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        stochastic gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        vs = []
        zs = [x]
        for b, w in zip(self.biases, self.weights): #forward:
            temp = np.dot(w, zs[-1]) + b
            vs.append(temp)
            zs.append(sigmoid(temp))

        db = [0] * (self.num_layers - 1)
        dw = [0] * (self.num_layers - 1)

        delta = self.loss_derivative_wr_output_activations(zs[-1], y) * sigmoid_derivative(vs[-1]) #last layer
        db[-1] = delta.sum(1).reshape([len(delta), 1])
        dw[-1] = np.dot(delta, zs[-2].transpose())

        for i in range(self.num_layers-3,-1,-1): #backward:
            delta = np.dot(self.weights[i+1].transpose(), delta) * sigmoid_derivative(vs[i])
            db[i] = delta.sum(1).reshape([len(delta), 1])
            dw[i] = np.dot(delta, zs[i].transpose())

        return (db, dw)

    def one_label_accuracy(self, data):
        """Return accuracy of network on data with numeric labels"""
        output_results = [(np.argmax(self.network_output_before_softmax(x)), y)
         for (x, y) in data]
        return sum(int(x == y) for (x, y) in output_results)/float(len(data))

    def one_hot_accuracy(self,data):
        """Return accuracy of network on data with one-hot labels"""
        output_results = [(np.argmax(self.network_output_before_softmax(x)), np.argmax(y))
                          for (x, y) in data]
        return sum(int(x == y) for (x, y) in output_results) / float(len(data))


    def network_output_before_softmax(self, x):
        """Return the output of the network before softmax if ``x`` is input."""
        layer = 0
        for b, w in zip(self.biases, self.weights):
            if layer == len(self.weights) - 1:
                x = np.dot(w, x) + b
            else:
                x = sigmoid(np.dot(w, x)+b)
            layer += 1
        return x

    def loss(self, data):
        """Return the loss of the network on the data"""
        loss_list = []
        for (x, y) in data:
            net_output_before_softmax = self.network_output_before_softmax(x)
            net_output_after_softmax = self.output_softmax(net_output_before_softmax)
            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(),y).flatten()[0])
        return sum(loss_list) / float(len(data))

    def output_softmax(self, output_activations):
        """Return output after softmax given output before softmax"""
        output_exp = np.exp(output_activations)
        return output_exp/output_exp.sum()

    def loss_derivative_wr_output_activations(self, output_activations, y):
        """Return derivative of loss with respect to the output activations before softmax"""
        return self.output_softmax(output_activations) - y


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
