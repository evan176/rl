#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    range = xrange
except:
    pass

import numpy
import six
import tensorflow as tf

from rl.mlp import weight_variable, bias_variable
from rl.utils import summarize_variable


def conv_net(channels, filters, poolings, width, height, depth=None,
             fc_dim=1024, alpha=0.2):
    """
    Create convolution network 
    Args:
        channels (list): a list contains channel size for each layers, including input layer
        filters (list): a list contains filter shape for each layers, ex:
            2D: [[height_patch_dim, width_patch_dim], ...]
            3D: [[depth_patch_dim, height_patch_dim, width_patch_dim], ...]
        poolings (list): a list contains pooling shape for each layers, ex:
            2D: [[height_pool_dim, width_pool_dim], ...]
            3D: [[depth_pool_dim, height_pool_dim, width_pool_dim], ...]
        width (int): image width
        height (int): image height
        depth (int): image depth (default: None)
        alpha (float): slope of LeakyReLU (default: 1e-3)
    Returns:
        network (tf.Tensor): output tensor, final layer is fully connected (?, 1024)
        input_x (tf.Tensor): input placeholder for network
        variables (dict): a dictionary contains all weight and bias variables
    Usage:
        # Create 2D cnn
        >>> channels = [3, 16, 32, 32] # input channel: 3 (RGB)
        >>> filters = [[5, 5], [5, 5], [3, 3]] # filter for each channel
        >>> poolings = [[2, 2], [2, 2], [2, 2]] # pooling shape for each channel
        >>> network, input_x, variables = conv_net(channels, filters, poolings, 1000, 500)
        >>> print(network)
        Tensor("Relu_3:0", shape=(?, 1024), dtype=float32)
        >>> print(input_x)
        Tensor("input_x:0", shape=(?, 768, 1024, 3), dtype=float32)
        >>> print(variables)
        {'conv_b_2': <tensorflow...>, 'conv_b_1': <tensorflow...>,
         'conv_w_1': <tensorflow...>, 'conv_w_2': <tensorflow...>,
         'conv_b_0': <tensorflow...>, 'conv_w_0': <tensorflow...>},
    """
    variables = {}

    # Create input holder for 2D or 3D image
    if depth:
        input_dims = [None, depth, height, width, channels[0]]
    else:
        input_dims = [None, height, width, channels[0]]
    input_x = tf.placeholder(tf.float32, input_dims, name="x")

    x = input_x
    for i in range(len(filters)):
        w_name = "conv_w_{}".format(i)
        b_name = "conv_b_{}".format(i)
        y_name = "conv_y_{}".format(i)
        act_y_name = "activate_{}".format(y_name)
        pool_name = "max_pool_{}".format(i)

        if depth:
            # Add 3D convolution layer
            w = weight_variable(
                [filters[i][0], filters[i][1], filters[i][2], channels[i], channels[i + 1]],
                name=w_name
            )
            b = bias_variable([channels[i + 1]], name=b_name)

            summarize_variable(w, w_name)
            summarize_variable(b, b_name)

            with tf.name_scope(y_name) as scope:
                y = tf.add(
                    tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
                    b, name=y_name
                )
                h_conv = tf.nn.leaky_relu(y, alpha, name=act_y_name)

            pooling_shape = [
                1, poolings[i][0], poolings[i][1], poolings[i][2], 1
            ]
            h_pool = tf.nn.max_pool3d(
                h_conv, ksize=pooling_shape, strides=pooling_shape,
                padding='SAME', name=pool_name
            )
        else:
            # Add 2D convolution layer
            w = weight_variable(
                [filters[i][0], filters[i][1], channels[i], channels[i + 1]],
                name=w_name
            )
            b = bias_variable([channels[i + 1]], name=b_name)

            summarize_variable(w, w_name)
            summarize_variable(b, b_name)

            with tf.name_scope(y_name) as scope:
                y = tf.add(
                    tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME'),
                    b, name=y_name
                )
                h_conv = tf.nn.leaky_relu(y, alpha, name=act_y_name)

            pooling_shape = [1, poolings[i][0], poolings[i][1], 1]
            h_pool = tf.nn.max_pool(
                h_conv, ksize=pooling_shape, strides=pooling_shape,
                padding='SAME', name=pool_name
            )
        variables[w_name], variables[b_name] = w, b
        x = h_pool

    # Compute flatten size
    flat_dim = 1
    for d in h_pool.shape.as_list()[1:]:
        flat_dim = flat_dim * d
    # Reshape h_pool to flattern tensor
    h_flat = tf.reshape(h_pool, [-1, flat_dim])

    w_name, b_name, y_name = "fc_w", "fc_b", "fc_y"
    act_y_name = "activate_{}".format(y_name)

    # Add fully connected layer
    variables[w_name] = weight_variable([flat_dim, fc_dim], name=w_name)
    variables[b_name] = bias_variable([fc_dim], name=b_name)

    summarize_variable(variables[w_name], w_name)
    summarize_variable(variables[b_name], b_name)

    with tf.name_scope(y_name) as scope:
        y = tf.add(tf.matmul(h_flat, variables[w_name]), variables[b_name], name=y_name)
        h_fc = tf.nn.leaky_relu(y, alpha, name=act_y_name)

    return h_fc, input_x, variables
