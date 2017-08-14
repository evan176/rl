#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy
import tensorflow as tf

from .agent import RLInterface
from .mlp import weight_variable, bias_variable, multilayer_perceptron
from .cnn import conv_net


class Actor(RLInterface):
    def __init__(self, session, network, x_input, network_vars,
                 fc_dim, output_dim, learning_rate=1e-4):
        super(Actor, self).__init__(session=session)

        self._x_input = x_input
        self._vars = network_vars
        self._fc_dim = fc_dim
        self._output_dim = output_dim
        self._learning_rate = learning_rate

        self._build_value(network)
        self._build_train_op()

        # Initial variables
        self._session.run(tf.global_variables_initializer())

    def train(self, x, chosen_action, td_error, learning_rate=None):
        if not learning_rate:
            learning_rate=self._learning_rate

        chosen_vector = self._create_action_vector(len(x), chosen_action)
        self._session.run(self._train_op, feed_dict={self._x_input: x,
            self._tderror: td_error, self._chosen_holder: chosen_vector,
            self._lr_input: learning_rate
        })

    def get_value(self, x):
        return self._session.run(self._output_value, feed_dict={
            self._x_input: x
        })

    def get_loss(self, x):
        pass

    def _build_value(self, network):
        with tf.variable_scope("Actor"):
            w = weight_variable([self._fc_dim, self._output_dim])
            b = bias_variable([self._output_dim])
            self._vars['out_w'] = w
            self._vars['out_b'] = b
            self._output_value = tf.nn.softmax(tf.add(
                tf.matmul(network, w), b, name="action_probability"
            ))

    def _build_train_op(self):
        with tf.variable_scope("Actor"):
            self._lr_input = tf.placeholder(tf.float32, shape=[])
            self._chosen_holder = tf.placeholder(
                tf.float32, shape=[None, self._output_dim]
            )
            self._tderror = tf.placeholder(tf.float32, shape=[None, 1])

            clip_value = tf.clip_by_value(self._output_value, 1e-20, 1.0)
            log_prob = tf.log(tf.reduce_sum(
                tf.multiply(clip_value, self._chosen_holder),
                axis=1, keep_dims=True
            ))

            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._lr_input
            )
            self._train_op = optimizer.minimize(
                -tf.multiply(log_prob, self._tderror), name="train"
            )

    def _create_action_vector(self, batch_size, next_available=None):
        if next_available:
            available_vector = numpy.zeros([batch_size, self._output_dim])
            for batch_index, row in enumerate(next_available):
                for column_index in row:
                    available_vector[batch_index][column_index] = 1
        else:
            available_vector = numpy.ones([batch_size, self._output_dim])
        return available_vector


class Critic(RLInterface):
    def __init__(self, session, network, x_input, network_vars,
                 fc_dim, output_dim=1, learning_rate=1e-4, discount=0.9):
        super(Critic, self).__init__(session=session)

        self._x_input = x_input
        self._vars = network_vars
        self._fc_dim = fc_dim
        self._output_dim = output_dim
        self._learning_rate = learning_rate
        self._discount = discount

        self._build_value(network)
        self._build_tderror_op()
        self._build_loss_op()
        self._build_train_op()

        # Initial variables
        self._session.run(tf.global_variables_initializer())

    def train(self, x, reward, next_x, done, learning_rate=None):
        if not learning_rate:
            learning_rate=self._learning_rate

        target_value = self._calculate_target_value(reward, next_x, done)

        self._session.run(self._train_op, feed_dict={
            self._x_input: x,
            self._target_holder: target_value ,
            self._lr_input: learning_rate
        })

    def get_value(self, x):
        return self._session.run(self._output_value, feed_dict={
            self._x_input: x
        })

    def get_tderror(self, x, reward, next_x, done):
        target_value = self._calculate_target_value(reward, next_x, done)
        return self._session.run(self._tderror, feed_dict={
            self._x_input: x,
            self._target_holder: target_value
        })

    def get_loss(self, x, reward, next_x, done):
        target_value = self._calculate_target_value(reward, next_x, done)
        return self._session.run(self._loss, feed_dict={
            self._x_input: x,
            self._target_holder: target_value
        })

    def _build_value(self, network):
        with tf.variable_scope("Critic"):
            w = weight_variable([self._fc_dim, self._output_dim])
            b = bias_variable([self._output_dim])
            self._vars['out_w'] = w
            self._vars['out_b'] = b
            self._output_value = tf.add(
                tf.matmul(network, w), b, name="state_value"
            )

    def _build_tderror_op(self):
        with tf.variable_scope("Critic"):
            self._target_holder = tf.placeholder(tf.float32, shape=[None, 1])
            self._tderror = tf.subtract(
                self._target_holder,
                self._output_value, name="td_error"
            )

    def _build_loss_op(self):
        with tf.variable_scope("Critic"):
            self._loss = tf.losses.huber_loss(
                self._target_holder, self._output_value
            )

    def _build_train_op(self):
        with tf.variable_scope("Critic"):
            self._lr_input = tf.placeholder(tf.float32, shape=[])
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._lr_input
            )
            self._train_op = optimizer.minimize(
                tf.reduce_mean(self._loss), name="train"
            )

    def _calculate_target_value(self, reward, next_x, done):
        # Get next value
        next_value = self.get_value(next_x)
        # Compute target value from next state
        target_value = []
        for i in range(len(reward)):
            if done[i][0]:
                target_value.append([reward[i][0]])
            else:
                target_value.append(
                    [reward[i][0] + self._discount * next_value[i][0]]
                )
        return target_value


class ActorCritic(RLInterface):

    def __init__(self, session, actor_network, actor_input, actor_vars,
                 actor_fc_dim, actor_output_dim, critic_network, critic_input,
                 critic_vars, critic_fc_dim, critic_output_dim,
                 learning_rate=1e-4, discount=0.9):
        super(ActorCritic, self).__init__(session=session)

        self._actor = Actor(
            session=self.session, network=actor_network, x_input=actor_input,
            network_vars=actor_vars, fc_dim=actor_fc_dim,
            output_dim=actor_output_dim, learning_rate=learning_rate
        )

        self._critic = Critic(
            session=self.session, network=critic_network, x_input=critic_input, 
            network_vars=critic_vars, fc_dim=critic_fc_dim,
            output_dim=critic_output_dim, learning_rate=learning_rate * 10,
            discount=discount
        )

    @classmethod
    def mlp(cls, session, dimensions, learning_rate=1e-4, discount=0.9):
        """
        Args:
            session (tf.Session): tensorflow session
            dimensions (list): a list contains all layers' dimensions,
                includin input & output action
            learning_rate (double):  learning rate for optimizer (default: 1e-4)
            discount (double): discount factor (default: 0.9)
        Returns:
            ActorCritic
        Usage:
            # Create 3 hidden layer perceptrons with input size: 30, action size: 10
            >>> agent = ActorCritic.mlp(tf.Session(), [30, 50, 50, 50, 10])
        """
        # Create target & evaluation network with multilayer perceptron
        actor_network, actor_input, actor_vars = multilayer_perceptron(dimensions[:-1])
        critic_network, critic_input, critic_vars = multilayer_perceptron(dimensions[:-1])
        return cls(
            session=session, actor_network=actor_network, actor_input=actor_input,
            actor_vars=actor_vars, actor_fc_dim=dimensions[-2],
            actor_output_dim=dimensions[-1], critic_input=critic_input,
            critic_network=critic_network, critic_vars=critic_vars,
            critic_fc_dim=dimensions[-2], critic_output_dim=1,
            learning_rate=learning_rate, discount=discount,
        )

    @classmethod
    def cnn(cls, session, action_dim, channels, filters, poolings,
            width, height, depth=None, fc_dim=1024, learning_rate=1e-4,
            discount=0.9):
        """
        Args:
            session (tf.Session): tensorflow session
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
            learning_rate (double):  learning rate for optimizer (default: 1e-4)
            discount (double): discount factor (default: 0.9)
        Returns:
            ActorCritic 
        Usage:
            >>> action_dim = 10 # Create 2D cnn
            >>> width, height = 1000, 500
            >>> channels = [3, 16, 32, 32] # input channel: 3 (RGB)
            >>> filters = [[5, 5], [5, 5], [3, 3]] # filter for each channel
            >>> poolings = [[2, 2], [2, 2], [2, 2]] # pooling shape for each channel
            >>> agent = ActorCritic.cnn(
                    tf.Session(), action_dim, channels, filters, poolings,
                    width, height
                )
        """
        actor_network, actor_input, actor_vars = conv_net(
            channels, filters, poolings, width, height, depth, fc_dim
        )
        critic_net, critic_input, critic_vars = conv_net(
            channels, filters, poolings, width, height, depth, fc_dim
        )
        return cls(
            session=session, actor_network=actor_network, actor_input=actor_input,
            actor_vars=actor_vars, actor_fc_dim=dimensions[-2],
            actor_output_dim=dimensions[-1], critic_input=critic_input,
            critic_network=critic_network, critic_vars=critic_vars,
            critic_fc_dim=dimensions[-2], critic_output_dim=1,
            learning_rate=learning_rate, discount=discount,
        )

    @property
    def actor(self):
        return self._actor

    @property
    def critic(self):
        return self._critic

    def train(self, x, action, reward, next_x, done, learning_rate=None):
        if not learning_rate:
            learning_rate = self._learning_rate
        tderror = self._critic.get_tderror(x, reward, next_x)
        self._critic.train(x, reward, next_x, done, learning_rate)
        self._actor.train(x, action, tderror, learning_rate)

    def get_value(self, x):
        return self._actor.get_value(x)

    def get_loss(self, x, reward, next_x):
        return self._critic.get_loss(x, reward, next_x, done)
