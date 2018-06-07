#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    range = xrange
except:
    pass

import six
import tensorflow as tf

from rl.utils import summarize_variable


def weight_variable(shape, name=None):
    """
    Create weight variable with "tf.Variable"
    Args:
        shape (list): shape of weight
        name (str): name of variable
    Returns:
        tf.Variable: weight variable
    Usage:
        >>> weight_variable([100, 100])
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def bias_variable(shape, name=None):
    """
    Create bias variable with "tf.Variable"
    Args:
        shape (list): shape of bias
        name (str): name of variable
    Returns:
        tf.Variable: bias variable
    Usage:
        >>> bias_variable([100])
    """
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


def multilayer_perceptron(dimensions, alpha=0.2):
    """
    Create multilayer perceptron
    Args:
        dimensions (list): dimensions of each layer, including
            input & final layer
        alpha (float): slope of LeakyReLU (default: 1e-3)
    Returns:
        network (tf.Tensor): network operation, output shape is same as last element in dimensions
        input_x (tf.Tensor): input placeholder for network
        variables (dict): a dictionary contains all weight and bias variables
    Usage:
        # Create multilayer perceptron with 2 hidden layers (20, 20)
        >>> network, input_x, variables = multilayer_perceptron([30, 20, 20 , 1])
        >>> print(network)
        Tensor("neuron_2:0", shape=(?, 1), dtype=float32)
        >>> print(input_x)
        Tensor("input_x:0", shape=(?, 30), dtype=float32)
        >>> print(variables)
        {'b_2': <tensorflow...>, 'b_1': <tensorflow...>, 'w_1': <tensorflow...>,
         'w_2': <tensorflow...>, 'b_0': <tensorflow...>, 'w_0': <tensorflow...>}
    """
    variables = {}

    input_x = tf.placeholder(tf.float32, [None, dimensions[0]], name="x")

    x = input_x
    for i in range(len(dimensions) - 1):
        w_name = "w_{}".format(i)
        b_name = "b_{}".format(i)
        y_name = "y_{}".format(i)
        act_y_name = "activate_{}".format(y_name)

        w = weight_variable([dimensions[i], dimensions[i + 1]], name=w_name)
        b = bias_variable([dimensions[i + 1]], name=b_name)
        variables[w_name] = w
        variables[b_name] = b
        summarize_variable(w, w_name)
        summarize_variable(b, b_name)

        with tf.name_scope(y_name) as scope:
            y = tf.add(tf.matmul(x, w), b, name=y_name)
            x = tf.nn.leaky_relu(y, alpha, name=act_y_name)

    network = x

    return network, input_x, variables
