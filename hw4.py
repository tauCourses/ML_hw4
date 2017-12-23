# coding=utf-8
from numpy import *
import os
import sys
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import matplotlib.pyplot as plt
import math

dir_path = repr(os.path.dirname(os.path.realpath(sys.argv[0]))).strip("'")

print 'Loading data set... ',
mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0, 8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_unscaled = data[train_idx[:6000], :].astype(float)
train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

validation_data_unscaled = data[train_idx[6000:], :].astype(float)
validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

test_data_unscaled = data[60000 + test_idx, :].astype(float)
test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)

print('done')


def sgd_svm_classifier_w(train_set_data, train_set_labels, eta0, C, T):
    w = numpy.zeros_like(train_data[0])
    for t in range(1, T + 1):
        eta_t = eta0 / t
        i = numpy.random.randint(0, len(train_set_data))
        value = numpy.dot(w, train_set_data[i]) * train_set_labels[i]
        if value < 1:
            change_w_by = C * train_set_labels[i] * \
                          (train_set_data[i] / numpy.linalg.norm(train_set_data[i]))
            w = w * (1 - eta_t) + eta_t * change_w_by
    w /= numpy.linalg.norm(w)  # normalize
    return (lambda x: (numpy.dot(w, x) > 0) * 2 - 1), w


def test_classifier(test_set_data, test_set_labels, classifier):
    accuracy = 0
    for i in range(len(test_set_data)):
        if classifier(test_set_data[i]) == test_set_labels[i]:
            accuracy += 1
    return float(accuracy) / len(test_set_data)


def assignment_1_a():
    print 'Running assignment 1a... ',
    T = 1000
    C = 1
    number_of_repeats = 10
    eta0_options = [math.pow(10, i) for i in numpy.linspace(-4, -1.5, 100)]
    train_result_accuracy = []
    validation_result_accuracy = []
    for eta0 in eta0_options:
        train_accuracy = 0
        validation_accuracy = 0
        for _ in range(number_of_repeats):
            classifier, w = sgd_svm_classifier_w(train_data, train_labels, eta0, C, T)
            train_accuracy += test_classifier(train_data, train_labels, classifier)
            validation_accuracy += test_classifier(validation_data, validation_labels, classifier)
        train_result_accuracy.append(train_accuracy / float(number_of_repeats))
        validation_result_accuracy.append(validation_accuracy / float(number_of_repeats))
    max_value = max(validation_result_accuracy)
    max_index = validation_result_accuracy.index(max_value)
    my_best_eta_0 = eta0_options[max_index]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.plot(eta0_options, train_result_accuracy, 'r-', label='Train Accuracy', )
    ax.plot(eta0_options, validation_result_accuracy, 'b-', label='Validation Accuracy')
    plt.legend()
    plt.xlabel('η0', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    fig.savefig(os.path.join(dir_path, '1_a.png'))
    fig.clf()
    print('done')
    print("Best η0 for SGD SVM was: " + str(my_best_eta_0))
    return my_best_eta_0


def assignment_1_b(eta0):
    print 'Running assignment 1b... ',
    T = 1000
    number_of_repeats = 10
    c_options = [math.pow(10, i) for i in numpy.linspace(-0.5, 0.5, 100)]
    train_result_accuracy = []
    validation_result_accuracy = []
    for C in c_options:
        train_accuracy = 0
        validation_accuracy = 0
        for _ in range(number_of_repeats):
            classifier, w = sgd_svm_classifier_w(train_data, train_labels, eta0, C, T)
            train_accuracy += test_classifier(train_data, train_labels, classifier)
            validation_accuracy += test_classifier(validation_data, validation_labels, classifier)
        train_result_accuracy.append(train_accuracy / float(number_of_repeats))
        validation_result_accuracy.append(validation_accuracy / float(number_of_repeats))
    max_value = max(validation_result_accuracy)
    max_index = validation_result_accuracy.index(max_value)
    my_best_c = c_options[max_index]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.plot(c_options, train_result_accuracy, 'r-', label='Train Accuracy', )
    ax.plot(c_options, validation_result_accuracy, 'b-', label='Validation Accuracy')
    plt.legend()
    plt.xlabel('C', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    fig.savefig(os.path.join(dir_path, '1_b.png'))
    fig.clf()
    print('done')
    print("Best C for SGD SVM was: " + str(my_best_c))
    return my_best_c


def assignment_1_c(eta0, C):
    print 'Running assignment 1c... ',
    T = 20000
    classifier, w = sgd_svm_classifier_w(train_data, train_labels, eta0, C, T)
    plt.imshow(reshape(w, (28, 28)), interpolation="nearest")
    plt.savefig(os.path.join(dir_path, '1_c.png'))
    print('done')
    return classifier


def assignment_1_d(classifier):
    print 'Running assignment 1d... ',
    accuracy = test_classifier(test_data, test_labels, classifier)
    print('done')
    print('accuracy of the best classifier was: ' + str(accuracy))


best_eta0 = assignment_1_a()
best_C = assignment_1_b(best_eta0)
best_classifier = assignment_1_c(best_eta0, best_C)
assignment_1_d(best_classifier)
