#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from unittest import mock
except:
    import mock
import numpy
import tensorflow as tf

from rl.mlp import (
    weight_variable, bias_variable, LeakyReLU, multilayer_perceptron
)


class WeightTest(tf.test.TestCase):

    def setUp(self):
        self._test_shape = [3, 3]

    def test_name(self):
        tf.reset_default_graph()
        with self.test_session():
            w = weight_variable(self._test_shape, "test_w")
            self.assertEqual(w.name, "test_w:0")

    def test_type(self):
        tf.reset_default_graph()
        with self.test_session():
            w = weight_variable(self._test_shape)
            self.assertIsInstance(w, tf.Variable)

    def test_shape(self):
        tf.reset_default_graph()
        with self.test_session():
            w = weight_variable(self._test_shape)
            self.assertAllEqual(w.get_shape(), self._test_shape)

    @mock.patch("tensorflow.truncated_normal")
    def test_value(self, mock_truncated_normal):
        def side_effect(shape, stddev):
            return numpy.ones(shape)
        mock_truncated_normal.side_effect = side_effect

        tf.reset_default_graph()
        with self.test_session() as sess:
            w = weight_variable(self._test_shape)
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(sess.run(w), numpy.ones(self._test_shape))


class BiasTest(tf.test.TestCase):

    def setUp(self):
        self._test_shape = [3, 3]

    def test_name(self):
        tf.reset_default_graph()
        with self.test_session():
            b = bias_variable(self._test_shape, "test_b")
            self.assertEqual(b.name, "test_b:0")

    def test_type(self):
        tf.reset_default_graph()
        with self.test_session():
            b = bias_variable(self._test_shape)
            self.assertIsInstance(b, tf.Variable)

    def test_shape(self):
        tf.reset_default_graph()
        with self.test_session():
            b = bias_variable(self._test_shape)
            self.assertAllEqual(b.get_shape(), self._test_shape)

    @mock.patch("tensorflow.constant")
    def test_value(self, mock_constant):
        def side_effect(value, shape):
            return numpy.ones(shape)
        mock_constant.side_effect = side_effect

        tf.reset_default_graph()
        with self.test_session() as sess:
            b = bias_variable(self._test_shape)
            sess.run(tf.global_variables_initializer())
            self.assertAllEqual(sess.run(b), numpy.ones(self._test_shape))


class LeakyReLUTest(tf.test.TestCase):

    def test_name(self):
        tf.reset_default_graph()
        with self.test_session():
            x = tf.placeholder(tf.float32, [1.0])
            a = LeakyReLU(x, 1, "test_relu")
            self.assertEqual(a.name, "test_relu:0")

    def test_type(self):
        tf.reset_default_graph()
        with self.test_session():
            x = tf.placeholder(tf.float32, [1.0])
            a = LeakyReLU(x, 1, "test_relu")
            self.assertIsInstance(a, tf.Tensor)

    def test_alpha_001(self):
        tf.reset_default_graph()
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, [1.0])
            a = LeakyReLU(x, 0.001, "test_relu")
            sess.run(tf.global_variables_initializer())
            self.assertEqual(sess.run(a, feed_dict={x: [10]}), 10)
            self.assertEqual(sess.run(a, feed_dict={x: [-1]}), -0.001)


class MLPTest(tf.test.TestCase):

    def setUp(self):
        self._test_dims = [5, 5, 5, 2]

    def test_input_name(self):
        tf.reset_default_graph()
        with self.test_session():
            with tf.name_scope("test"):
                net, input_x, m_vars = multilayer_perceptron(self._test_dims)
                self.assertEqual(input_x.name, "test/x:0")

    def test_input_type(self):
        tf.reset_default_graph()
        with self.test_session():
            with tf.name_scope("test"):
                net, input_x, m_vars = multilayer_perceptron(self._test_dims)
                self.assertIsInstance(input_x, tf.Tensor)

    def test_input_shape(self):
        tf.reset_default_graph()
        with self.test_session():
            with tf.name_scope("test"):
                net, input_x, m_vars = multilayer_perceptron(self._test_dims)
                self.assertAllEqual(
                    [x.value for x in input_x.get_shape().dims],
                    [None, self._test_dims[0]]
                )

    def test_vars_name(self):
        tf.reset_default_graph()
        with self.test_session():
            with tf.name_scope("test"):
                net, input_x, m_vars = multilayer_perceptron(self._test_dims)
                for i in range(len(self._test_dims) - 1):
                    w_name, b_name = 'w_{}'.format(i), 'b_{}'.format(i)
                    w, b = m_vars[w_name], m_vars[b_name]

                    self.assertEqual(
                        w.name, "test/{}:0".format(w_name)
                    )
                    self.assertEqual(
                        b.name, "test/{}:0".format(b_name)
                    )

    def test_vars_type(self):
        tf.reset_default_graph()
        with self.test_session():
            with tf.name_scope("test"):
                net, input_x, m_vars = multilayer_perceptron(self._test_dims)
                for i in range(len(self._test_dims) - 1):
                    w_name, b_name = 'w_{}'.format(i), 'b_{}'.format(i)
                    w, b = m_vars[w_name], m_vars[b_name]

                    self.assertIsInstance(w, tf.Variable)
                    self.assertIsInstance(b, tf.Variable)

    def test_vars_shape(self):
        tf.reset_default_graph()
        with self.test_session():
            with tf.name_scope("test"):
                net, input_x, m_vars = multilayer_perceptron(self._test_dims)
                for i in range(len(self._test_dims) - 1):
                    w_name, b_name = 'w_{}'.format(i), 'b_{}'.format(i)
                    w, b = m_vars[w_name], m_vars[b_name]

                    self.assertAllEqual(
                        [x.value for x in w.get_shape().dims],
                        [self._test_dims[i], self._test_dims[i + 1]]
                    )
                    self.assertAllEqual(
                        [x.value for x in b.get_shape().dims],
                        [self._test_dims[i + 1]]
                    )

    def test_net_name(self):
        tf.reset_default_graph()
        with self.test_session():
            with tf.name_scope("test"):
                net, input_x, m_vars = multilayer_perceptron(self._test_dims)
                dim_len = len(self._test_dims)
                act_name = "test/y_{}/activate_y_{}:0".format(
                    dim_len - 2, dim_len - 2
                )
                self.assertEqual(net.name, act_name)

    def test_net_type(self):
        tf.reset_default_graph()
        with self.test_session():
            with tf.name_scope("test"):
                net, input_x, m_vars = multilayer_perceptron(self._test_dims)
                self.assertIsInstance(net, tf.Tensor)

    def test_net_shape(self):
        tf.reset_default_graph()
        with self.test_session():
            with tf.name_scope("test"):
                net, input_x, m_vars = multilayer_perceptron(self._test_dims)
                self.assertEqual(
                    [x.value for x in net.get_shape().dims],
                    [None, self._test_dims[-1]]
                )
                # [x.value for x in net.get_shape().dims],
