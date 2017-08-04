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

from rl.mlp import weight_variable, bias_variable, multilayer_perceptron


class MLPTest(TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @mock.patch("tensorflow.truncated_normal")
    def test_weight(self, mock_truncated_normal):
        mock_truncated_normal.side_effect = lambda x, **kargs: numpy.ones(x)

        w = weight_variable([3, 3])
        self.assertIsInstance(w, tf.Variable)
        self.assertEqual(w.get_shape(), [3, 3])

    @mock.patch("tensorflow.constant")
    def test_bias(self, mock_constant):
        mock_constant.side_effect = lambda x, **kargs: numpy.ones(kargs['shape'])

        b = bias_variable([3, 3])
        self.assertIsInstance(b, tf.Variable)
        self.assertEqual(b.get_shape(), [3, 3])

    def test_mlp(self):
        test_dimensions = [5, 5, 5, 2]
        net, input_x, variables = multilayer_perceptron(test_dimensions)

        self.assertIsInstance(net, tf.Tensor)
        self.assertEqual([x.value for x in net.get_shape().dims], [None, test_dimensions[-1]])

        self.assertIsInstance(input_x, tf.Tensor)
        self.assertEqual([x.value for x in input_x.get_shape().dims], [None, test_dimensions[0]])

        for i in range(len(test_dimensions) - 1):
            w = variables['w_{}'.format(i)]
            b = variables['b_{}'.format(i)]
            self.assertEqual(w.get_shape(), [test_dimensions[i], test_dimensions[i + 1]])
            self.assertEqual(b.get_shape(), [test_dimensions[i + 1]])
