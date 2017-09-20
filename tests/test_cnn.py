#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    range = xrange
except:
    pass

try:
    from unittest import mock
except:
    import mock

import numpy
import six
import tensorflow as tf

from rl.cnn import conv_net


class CNN2DTest(tf.test.TestCase):

    def setUp(self):
        self._channels = [3, 20, 30, 40]
        self._filters = [[5, 5], [4, 4], [3, 3]]
        self._poolings = [[4, 4], [3, 3], [2, 2]]
        self._width = 1000
        self._height = 1000
        self._depth = None
        self._fc_dim = 1024
        self._alpha = 1e-3
        self._test_scope = "test"

        tf.reset_default_graph()
        with self.test_session() as sess:
            with tf.name_scope(self._test_scope):
                self._net, self._input_x, self._vars = conv_net(
                    self._channels, self._filters, self._poolings,
                    self._width, self._height, self._depth,
                    self._fc_dim, self._alpha
                )

    def test_input_name(self):
        self.assertEqual(self._input_x.name, "{}/x:0".format(self._test_scope))

    def test_input_type(self):
        self.assertIsInstance(self._input_x, tf.Tensor)

    def test_input_shape(self):
        self.assertAllEqual(
            [x.value for x in self._input_x.get_shape().dims],
            [None, self._height, self._width, self._channels[0]]
        )

    def test_vars_name(self):
        for i in range(len(self._channels) - 1):
            w_name = 'conv_w_{}'.format(i)
            b_name = 'conv_b_{}'.format(i)
            w, b = self._vars[w_name], self._vars[b_name]

            self.assertEqual(
                w.name, "{}/{}:0".format(self._test_scope, w_name)
            )
            self.assertEqual(
                b.name, "{}/{}:0".format(self._test_scope, b_name)
            )

        self.assertEqual(
            self._vars['fc_w'].name, "{}/fc_w:0".format(self._test_scope)
        )
        self.assertEqual(
            self._vars['fc_b'].name, "{}/fc_b:0".format(self._test_scope)
        )

    def test_vars_type(self):
        for i in range(len(self._channels) - 1):
            w_name = 'conv_w_{}'.format(i)
            b_name = 'conv_b_{}'.format(i)
            w, b = self._vars[w_name], self._vars[b_name]

            self.assertIsInstance(w, tf.Variable)
            self.assertIsInstance(b, tf.Variable)

        self.assertIsInstance(self._vars['fc_w'], tf.Variable)
        self.assertIsInstance(self._vars['fc_b'], tf.Variable)

    def test_vars_shape(self):
        for i in range(len(self._channels) - 1):
            w_name = 'conv_w_{}'.format(i)
            b_name = 'conv_b_{}'.format(i)
            w, b = self._vars[w_name], self._vars[b_name]

            self.assertIsInstance(w, tf.Variable)
            self.assertIsInstance(b, tf.Variable)

            self.assertAllEqual(
                [x.value for x in w.get_shape().dims],
                [*self._filters[i], self._channels[i], self._channels[i + 1]]
            )
            self.assertAllEqual(
                [x.value for x in b.get_shape().dims],
                [self._channels[i + 1]]
            )

    def test_net_name(self):
        self.assertEqual(
            self._net.name, "{}/fc_y/activate_fc_y:0".format(self._test_scope)
        )

    def test_net_type(self):
        self.assertIsInstance(self._net, tf.Tensor)

    def test_net_shape(self):
        self.assertAllEqual(
            [x.value for x in self._net.get_shape().dims],
            [None, self._fc_dim]
        )

class CNN3DTest(CNN2DTest):
    def setUp(self):
        self._channels = [10, 20, 30, 40]
        self._filters = [[1, 5, 5], [1, 4, 4], [1, 3, 3]]
        self._poolings = [[1, 4, 4], [1, 3, 3], [1, 2, 2]]
        self._width = 500
        self._height = 500
        self._depth = 10
        self._fc_dim = 1024
        self._alpha = 1e-3
        self._test_scope = "test"

        tf.reset_default_graph()
        with self.test_session() as sess:
            with tf.name_scope(self._test_scope):
                self._net, self._input_x, self._vars = conv_net(
                    self._channels, self._filters, self._poolings,
                    self._width, self._height, self._depth,
                    self._fc_dim, self._alpha
                )

    def test_input_shape(self):
        self.assertAllEqual(
            [x.value for x in self._input_x.get_shape().dims],
            [None, self._depth, self._height, self._width, self._channels[0]]
        )
