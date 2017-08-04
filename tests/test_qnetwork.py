#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import copy
import random
from types import GeneratorType

try:
    from unittest import mock
except:
    import mock
import numpy
import tensorflow as tf

from rl.qnetwork import DQN


class DQNMLPTest(tf.test.TestCase):

    def test_init(self):
        test_dims = [30, 20, 10, 5]
        with self.test_session() as sess:
            agent = DQN.mlp(sess, test_dims)
            # Check evaluation network & target output's type & shape
            test_shape = [None, test_dims[-1]]
            self.assertIsInstance(agent._eval_q, tf.Tensor)
            self.assertEqual(agent._eval_q.shape.as_list(), test_shape)
            self.assertIsInstance(agent._target_q, tf.Tensor)
            self.assertEqual(agent._target_q.shape.as_list(), test_shape)
            # Check evaluation network & target input's type & shape
            test_shape = [None, test_dims[0]]
            self.assertIsInstance(agent._eval_input, tf.Tensor)
            self.assertEqual(
                agent._eval_input.shape.as_list(), test_shape
            )
            self.assertIsInstance(agent._target_input, tf.Tensor)
            self.assertEqual(
                agent._target_input.shape.as_list(), test_shape
            )
            # Check evaluation network & target network's type & shape
            test_shape = [None, test_dims[-2]]
            self.assertIsInstance(agent._eval_network, tf.Tensor)
            self.assertEqual(
                agent._eval_network.shape.as_list(), test_shape
            )
            self.assertIsInstance(agent._target_network, tf.Tensor)
            self.assertEqual(
                agent._target_network.shape.as_list(), test_shape
            )

    def test_train(self):
        test_dims = [30, 20, 10, 5]
        batch_size = 100

        with self.test_session() as sess:
            agent = DQN.mlp(sess, test_dims, replace_iter=3)

            x = numpy.random.rand(batch_size, test_dims[0])
            chosen_action = [[random.randint(0, test_dims[-1] - 1)]
                             for i in range(batch_size)]
            reward = numpy.random.rand(batch_size, 1)
            next_x = numpy.random.rand(batch_size, test_dims[0])
            next_available = []
            for i in range(batch_size):
                num = random.randint(0, test_dims[-1] - 1)
                next_available.append(random.sample(range(test_dims[-1]), num))

            returns = agent.train(
                x, chosen_action, reward, next_x, next_available
            )
            # Check replace count
            self.assertEqual(agent._train_step, 1)

            for i in range(2):
                returns = agent.train(
                    x, chosen_action, reward, next_x, next_available
                )
            self.assertEqual(agent._train_step, 3)

    def test_get_loss(self):
        test_dims = [30, 20, 10, 5]
        batch_size = 100

        with self.test_session() as sess:
            agent = DQN.mlp(sess, test_dims)

            x = numpy.random.rand(batch_size, test_dims[0])
            chosen_action = [[random.randint(0, test_dims[-1] - 1)]
                             for i in range(batch_size)]
            reward = numpy.random.rand(batch_size, 1)
            next_x = numpy.random.rand(batch_size, test_dims[0])
            next_available = []
            for i in range(batch_size):
                num = random.randint(0, test_dims[-1] - 1)
                next_available.append(random.sample(range(test_dims[-1]), num))
            # Check loss's shape
            returns = agent.get_loss(
                x, chosen_action, reward, next_x, next_available
            )
            self.assertEqual(returns.shape, (batch_size, 1))

    def test_getQ(self):
        test_dims = [30, 20, 10, 5]
        batch_size = 1000

        with self.test_session() as sess:
            agent = DQN.mlp(sess, test_dims)

            returns = agent.get_value(
                numpy.random.rand(batch_size, test_dims[0])
            )
            # Check Q's shape
            self.assertEqual(returns.shape, (batch_size, test_dims[-1]))


class DQNCNN2DTest(tf.test.TestCase):

    def test_init(self):
        channels = [3, 16, 32]
        filters = [[5, 5], [3, 3]]
        poolings = [[2, 2], [3, 3]]
        action_size = 5
        width, height = 100, 50

        with self.test_session() as sess:
            agent = DQN.cnn(
                sess, action_size, channels, filters, poolings, width, height
            )

            test_shape = [None, action_size]
            self.assertIsInstance(agent._eval_q, tf.Tensor)
            self.assertEqual(agent._eval_q.shape.as_list(), test_shape)
            self.assertIsInstance(agent._target_q, tf.Tensor)
            self.assertEqual(agent._target_q.shape.as_list(), test_shape)

            test_shape = [None, height, width, channels[0]]
            self.assertIsInstance(agent._eval_input, tf.Tensor)
            self.assertEqual(agent._eval_input.shape.as_list(), test_shape)
            self.assertIsInstance(agent._target_input, tf.Tensor)
            self.assertEqual(agent._target_input.shape.as_list(), test_shape)

            test_shape = [None, 1024]
            self.assertIsInstance(agent._eval_network, tf.Tensor)
            self.assertEqual(agent._eval_network.shape.as_list(), test_shape)
            self.assertIsInstance(agent._target_network, tf.Tensor)
            self.assertEqual(agent._target_network.shape.as_list(), test_shape)

    def test_train(self):
        channels = [3, 16, 32]
        filters = [[5, 5], [3, 3]]
        poolings = [[2, 2], [3, 3]]
        action_size = 5
        width, height = 100, 50
        batch_size = 50

        with self.test_session() as sess:
            agent = DQN.cnn(
                sess, action_size, channels, filters, poolings, width, height,
                replace_iter=3
            )

            x = numpy.random.rand(batch_size, height, width, 3)
            chosen_action = [[random.randint(0, action_size - 1)]
                             for i in range(batch_size)]
            reward = numpy.random.rand(batch_size, 1)
            next_x = numpy.random.rand(batch_size, height, width, 3)
            next_available = []
            for i in range(batch_size):
                num = random.randint(0, action_size - 1)
                next_available.append(random.sample(range(action_size), num))

            returns = agent.train(
                x, chosen_action, reward, next_x, next_available
            )
            # Check replace count
            self.assertEqual(agent._train_step, 1)
            for i in range(2):
                returns = agent.train(
                    x, chosen_action, reward, next_x, next_available
                )
            self.assertEqual(agent._train_step, 3)

    def test_get_loss(self):
        channels = [3, 16, 32]
        filters = [[5, 5], [3, 3]]
        poolings = [[2, 2], [3, 3]]
        action_size = 5
        width, height = 100, 50
        batch_size = 50

        with self.test_session() as sess:
            agent = DQN.cnn(
                sess, action_size, channels, filters, poolings, width, height
            )

            x = numpy.random.rand(
                batch_size, height, width, channels[0]
            )
            chosen_action = [[random.randint(0, action_size - 1)]
                             for i in range(batch_size)]
            reward = numpy.random.rand(batch_size, 1)
            next_x = numpy.random.rand(
                batch_size, height, width, channels[0]
            )
            next_available = []
            for i in range(batch_size):
                num = random.randint(0, action_size - 1)
                next_available.append(random.sample(range(action_size), num))

            # Check loss's shape
            returns = agent.get_loss(
                x, chosen_action, reward, next_x, next_available
            )
            self.assertEqual(returns.shape, (batch_size, 1))

    def test_getQ(self):
        channels = [3, 16, 32]
        filters = [[5, 5], [3, 3]]
        poolings = [[2, 2], [3, 3]]
        action_size = 5
        width, height = 100, 50
        batch_size = 10

        with self.test_session() as sess:
            agent = DQN.cnn(
                sess, action_size, channels, filters, poolings, width, height
            )

            returns = agent.get_value(
                numpy.random.rand(batch_size, height, width, channels[0])
            )
            # Check Q's shape
            self.assertEqual(returns.shape, (batch_size, action_size))


class DQNCNN3DTest(tf.test.TestCase):

    def test_init(self):
        channels = [3, 16, 32]
        filters = [[1, 5, 5], [1, 3, 3]]
        poolings = [[1, 2, 2], [1, 3, 3]]
        action_size = 5
        width, height, depth = 50, 20, 5

        with self.test_session() as sess:
            agent = DQN.cnn(
                sess, action_size, channels, filters, poolings,
                width, height, depth
            )
            test_shape = [None, action_size]
            self.assertIsInstance(agent._eval_q, tf.Tensor)
            self.assertEqual(agent._eval_q.shape.as_list(), test_shape)
            self.assertIsInstance(agent._target_q, tf.Tensor)
            self.assertEqual(agent._target_q.shape.as_list(), test_shape)

            test_shape = [None, depth, height, width, channels[0]]
            self.assertIsInstance(agent._eval_input, tf.Tensor)
            self.assertEqual(agent._eval_input.shape.as_list(), test_shape)
            self.assertIsInstance(agent._target_input, tf.Tensor)
            self.assertEqual(agent._target_input.shape.as_list(), test_shape)

            test_shape = [None, 1024]
            self.assertIsInstance(agent._eval_network, tf.Tensor)
            self.assertEqual(agent._eval_network.shape.as_list(), test_shape)
            self.assertIsInstance(agent._target_network, tf.Tensor)
            self.assertEqual(agent._target_network.shape.as_list(), test_shape)

    def test_train(self):
        channels = [3, 16, 32]
        filters = [[1, 5, 5], [1, 3, 3]]
        poolings = [[1, 2, 2], [2, 3, 3]]
        action_size = 5
        width, height, depth = 50, 20, 5
        batch_size = 10

        with self.test_session() as sess:
            agent = DQN.cnn(
                sess, action_size, channels, filters, poolings,
                width, height, depth, replace_iter=3
            )

            x = numpy.random.rand(
                batch_size, depth, height, width, channels[0]
            )
            chosen_action = [[random.randint(0, action_size - 1)]
                             for i in range(batch_size)]
            reward = numpy.random.rand(batch_size, 1)
            next_x = numpy.random.rand(
                batch_size, depth, height, width, channels[0]
            )
            next_available = []
            for i in range(batch_size):
                num = random.randint(0, action_size - 1)
                next_available.append(random.sample(range(action_size), num))

            returns = agent.train(
                x, chosen_action, reward, next_x, next_available
            )
            # Check replace count
            self.assertEqual(agent._train_step, 1)
            for i in range(2):
                returns = agent.train(
                    x, chosen_action, reward, next_x, next_available
                )
            self.assertEqual(agent._train_step, 3)

    def test_get_loss(self):
        channels = [3, 16, 32]
        filters = [[1, 5, 5], [2, 3, 3]]
        poolings = [[3, 2, 2], [1, 3, 3]]
        action_size = 5
        width, height, depth = 50, 20, 5
        batch_size = 10

        with self.test_session() as sess:
            agent = DQN.cnn(
                sess, action_size, channels, filters, poolings,
                width, height, depth
            )

            x = numpy.random.rand(
                batch_size, depth, height, width, channels[0]
            )
            chosen_action = [[random.randint(0, action_size - 1)]
                             for i in range(batch_size)]
            reward = numpy.random.rand(batch_size, 1)
            next_x = numpy.random.rand(
                batch_size, depth, height, width, channels[0]
            )
            next_available = []
            for i in range(batch_size):
                num = random.randint(0, action_size - 1)
                next_available.append(random.sample(range(action_size), num))

            # Check loss's shape
            returns = agent.get_loss(
                x, chosen_action, reward, next_x, next_available
            )
            self.assertEqual(returns.shape, (batch_size, 1))

    def test_getQ(self):
        channels = [3, 16, 32]
        filters = [[1, 5, 5], [1, 3, 3]]
        poolings = [[1, 2, 2], [1, 3, 3]]
        action_size = 5
        width, height, depth = 50, 20, 5
        batch_size = 10

        with self.test_session() as sess:
            agent = DQN.cnn(
                sess, action_size, channels, filters, poolings,
                width, height, depth
            )

            returns = agent.get_value(
                numpy.random.rand(batch_size, depth, height, width, channels[0])
            )
            # Check Q's shape
            self.assertEqual(returns.shape, (batch_size, action_size))
