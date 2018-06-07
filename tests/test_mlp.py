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

from rl.mlp import (
    weight_variable, bias_variable, multilayer_perceptron
)


class WeightTest(tf.test.TestCase):

    @mock.patch("tensorflow.truncated_normal")
    def setUp(self, mock_truncated_normal):
        def side_effect(shape, stddev):
            return numpy.ones(shape)
        mock_truncated_normal.side_effect = side_effect

        self._test_shape = [3, 3]
        self._test_name = "test_w"

        tf.reset_default_graph()
        with self.test_session() as sess:
            self._w = weight_variable(self._test_shape, self._test_name)
            self._sess = sess

    def test_name(self):
        self.assertEqual(self._w.name, "{}:0".format(self._test_name))

    def test_type(self):
        self.assertIsInstance(self._w, tf.Variable)

    def test_shape(self):
        self.assertAllEqual(self._w.get_shape(), self._test_shape)

    def test_value(self):
        self._sess.run(tf.global_variables_initializer())
        self.assertAllEqual(self._sess.run(self._w), numpy.ones(self._test_shape))


class BiasTest(tf.test.TestCase):

    @mock.patch("tensorflow.constant")
    def setUp(self, mock_constant):
        def side_effect(_, shape):
            return numpy.ones(shape)
        mock_constant.side_effect = side_effect

        self._test_shape = [3, 3]
        self._test_name = "test_b"

        tf.reset_default_graph()
        with self.test_session() as sess:
            self._sess = sess
            self._b = bias_variable(self._test_shape, self._test_name)

    def test_name(self):
        self.assertEqual(self._b.name, "{}:0".format(self._test_name))

    def test_type(self):
        self.assertIsInstance(self._b, tf.Variable)

    def test_shape(self):
        self.assertAllEqual(self._b.get_shape(), self._test_shape)

    def test_value(self):
        self._sess.run(tf.global_variables_initializer())
        self.assertAllEqual(self._sess.run(self._b), numpy.ones(self._test_shape))


class MLPTest(tf.test.TestCase):

    def setUp(self):
        self._test_dims = [5, 5, 5, 2]
        self._test_scope = "test"

        tf.reset_default_graph()
        with self.test_session() as sess:
            self._sess = sess
            with tf.name_scope(self._test_scope):
                self._net, self._input_x, self._vars = multilayer_perceptron(self._test_dims)

    def test_input_name(self):
        self.assertEqual(self._input_x.name, "{}/x:0".format(self._test_scope))

    def test_input_type(self):
        self.assertIsInstance(self._input_x, tf.Tensor)

    def test_input_shape(self):
        self.assertAllEqual(
            [x.value for x in self._input_x.get_shape().dims],
            [None, self._test_dims[0]]
        )

    def test_vars_name(self):
        for i in range(len(self._test_dims) - 1):
            w_name, b_name = 'w_{}'.format(i), 'b_{}'.format(i)
            w, b = self._vars[w_name], self._vars[b_name]

            self.assertEqual(
                w.name, "{}/{}:0".format(self._test_scope, w_name)
            )
            self.assertEqual(
                b.name, "{}/{}:0".format(self._test_scope, b_name)
            )

    def test_vars_type(self):
        for i in range(len(self._test_dims) - 1):
            w_name, b_name = 'w_{}'.format(i), 'b_{}'.format(i)
            w, b = self._vars[w_name], self._vars[b_name]

            self.assertIsInstance(w, tf.Variable)
            self.assertIsInstance(b, tf.Variable)

    def test_vars_shape(self):
        for i in range(len(self._test_dims) - 1):
            w_name, b_name = 'w_{}'.format(i), 'b_{}'.format(i)
            w, b = self._vars[w_name], self._vars[b_name]

            self.assertAllEqual(
                [x.value for x in w.get_shape().dims],
                [self._test_dims[i], self._test_dims[i + 1]]
            )
            self.assertAllEqual(
                [x.value for x in b.get_shape().dims],
                [self._test_dims[i + 1]]
            )

    def test_net_name(self):
        dim_len = len(self._test_dims)
        act_name = "{}/y_{}/activate_y_{}:0".format(
            self._test_scope, dim_len - 2, dim_len - 2
        )
        self.assertEqual(self._net.name, act_name)

    def test_net_type(self):
        self.assertIsInstance(self._net, tf.Tensor)

    def test_net_shape(self):
        self.assertEqual(
            [x.value for x in self._net.get_shape().dims],
            [None, self._test_dims[-1]]
        )
