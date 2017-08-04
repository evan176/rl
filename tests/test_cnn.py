#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import copy
from types import GeneratorType
from unittest import TestCase

try:
    from unittest import mock
except:
    import mock
import numpy
import tensorflow as tf

from rl.cnn import conv_net


class CNNTest(TestCase):

    def test_2d(self):
        # Init test
        channels = [3, 20, 30, 40]
        filters = [[5, 5], [4, 4], [3, 3]]
        poolings = [[4, 4], [3, 3], [2, 2]]
        width = 1000
        height = 1000
        net, input_x, variables = conv_net(
            channels, filters, poolings, width, height
        )
        # Check type and dimension of net
        self.assertIsInstance(net, tf.Tensor)
        self.assertEqual([x.value for x in net.get_shape().dims], [None, 1024])
        # Check type and dimension of input_x
        self.assertIsInstance(input_x, tf.Tensor)
        dims = [x.value for x in input_x.get_shape().dims]
        expected = [None, height, width, 3]
        self.assertEqual(dims, expected)
        # Check dimension of each variables
        for i in range(len(filters)):
            w = variables['conv_w_{}'.format(i)]
            b = variables['conv_b_{}'.format(i)]
            expected = [
                filters[i][0], filters[i][1], channels[i], channels[i + 1]
            ]
            self.assertEqual(w.get_shape(), expected)
            self.assertEqual(b.get_shape(), [channels[i + 1]])

    def test_3d(self):
        channels = [10, 20, 30, 40]
        filters = [[1, 5, 5], [1, 4, 4], [1, 3, 3]]
        poolings = [[1, 4, 4], [1, 3, 3], [1, 2, 2]]
        width = 1000
        height = 1000
        depth = 20
        net, input_x, variables = conv_net(
            channels, filters, poolings, width, height, depth
        )
        # Check type and dimension of net
        self.assertIsInstance(net, tf.Tensor)
        self.assertEqual([x.value for x in net.get_shape().dims], [None, 1024])
        # Check type and dimension of input_x
        self.assertIsInstance(input_x, tf.Tensor)
        dims = [x.value for x in input_x.get_shape().dims]
        expected = [None, depth, height, width, 10]
        self.assertEqual(dims, expected)
        # Check dimension of each variables
        for i in range(len(filters)):
            w = variables['conv_w_{}'.format(i)]
            b = variables['conv_b_{}'.format(i)]
            expected = [
                filters[i][0], filters[i][1], filters[i][2],
                channels[i], channels[i + 1]
            ]
            self.assertEqual(w.get_shape(), expected)
            self.assertEqual(b.get_shape(), [channels[i + 1]])
