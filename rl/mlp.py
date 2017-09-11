#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

from .utils import summarize_variable


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
    return tf.Variable(tf.truncated_normal(shape, stddev=0.001), name=name)


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


def normalize_weight(w, dim, name=None, epsilon=1e-12):
    """
    Create weight normalization op
    Args:
        w (tf.Variable): weight variable
        b (tf.Variable): bias variable
        dim (int): dimension for reduce_sum
    Returns:
        norm_w (tf.Tensor): normalized weight
        norm_b (tf.Tensor): normalized bias
    Usage:
        >>> norm_w = normalize_weight(w, 0)
    """
    with tf.name_scope(name) as scope:
        square_sum = tf.reduce_sum(tf.square(w), dim, keep_dims=True)
        inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        norm_w = tf.multiply(w, inv_norm)
    return norm_w


def LeakyReLU(x, alpha, name=None):
    with tf.name_scope(name) as scope:
        return tf.maximum(alpha * x, x, name=name)


def multilayer_perceptron(dimensions, alpha=1e-3):
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

    input_x = tf.placeholder(tf.float32, [None, dimensions[0]], name="input_x")

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
        # norm_w = normalize_weight(w, 0, "norm_{}".format(i))

        with tf.name_scope(y_name) as scope:
            # y = tf.add(tf.matmul(x, norm_w), b, name=y_name)
            y = tf.add(tf.matmul(x, w), b, name=y_name)

        x = LeakyReLU(y, alpha, name=act_y_name)
    network = x

    return network, input_x, variables
