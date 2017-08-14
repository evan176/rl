#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy
import tensorflow as tf

from .agent import RLInterface
from .mlp import (
    weight_variable, bias_variable, normalize_weight, multilayer_perceptron
)
from .cnn import conv_net
from .utils import summarize_variable


class DQN(RLInterface):
    def __init__(self, session, eval_network, target_network, eval_input,
                 target_input, eval_vars, target_vars, fc_dim,
                 action_dim, learning_rate=1e-4, discount=0.9,
                 replace_iter=100, alpha=1e-3, logdir=None):
        """
        Args:
            session (tf.Session): tensorflow session
            eval_network (tf.Tensor): evaluation network, last layer must be fully connected.
            target_network (tf.Tensor): target network, shape of each layer must be same as eval_network.
            eval_input (tf.Tensor): input placeholder for evaluation network
            target_input (tf.Tensor): input placeholder for target network
            eval_vars (dict): a dictionary contains all evaluation variables.
            target_vars (dict): a dictionary contains all target variables.
                Name of keys in target_vars must be same as eval_vars
            fc_dim (int): fully connected layer size
            action_dim (int): action layer (output) size
            learning_rate (double):  learning rate for optimizer (default: 1e-4)
            discount (double): discount factor (default: 0.9)
            replace_iter (int): replace iteration (default: 100)
            alpha (float): slope of LeakyReLU (default: 1e-3)
        Returns:
        Usage:
            # Init DQN with multilayer perceptron
            # 2 hidden layers input size: 30, output action: 5
            >>> agent = DQN.mlp(tf.Session(), [30, 50, 50, 5])
            # Init random data
            >>> x = numpy.random.rand(10, 30)
            # Init chosen action list, element in list is index of chosen action
            >>> chosen_action = [[3], [1], ... [4]]
            # Reward value for each records
            >>> reward = [[10], [0], ... [0]]
            >>> next_x = numpy.random.rand(10, 30)
            >>> agent.train(x, chosen_action, reward, next_x)
        """
        super(DQN, self).__init__(session=session)
        self._eval_network = eval_network
        self._target_network = target_network
        self._eval_vars = eval_vars
        self._target_vars = target_vars
        self._eval_input = eval_input
        self._target_input = target_input
        self._fc_dim = fc_dim
        self._action_dim = action_dim
        self._learning_rate = learning_rate
        self._discount = discount
        self._replace_iter = replace_iter
        self._train_step = 0

        self._build_q()
        self._build_tderror()
        self._build_train_op()
        self._build_replace_op()

        self._merged = tf.summary.merge_all()
        if logdir:
            self._train_writer = tf.summary.FileWriter(logdir, self.graph)
        else:
            self._train_writer = None
        # Initial variables
        self._session.run(tf.global_variables_initializer())

    @classmethod
    def mlp(cls, session, dimensions, learning_rate=1e-4, discount=0.9,
            replace_iter=100, alpha=1e-3, logdir=None):
        """
        Args:
            session (tf.Session): tensorflow session
            dimensions (list): a list contains all layers' dimensions,
                includin input & output action
            learning_rate (double):  learning rate for optimizer (default: 1e-4)
            discount (double): discount factor (default: 0.9)
            replace_iter (int): replace iteration (default: 100)
            alpha (float): slope of LeakyReLU (default: 1e-3)
        Returns:
            DQN
        Usage:
            # Create 3 hidden layer perceptrons with input size: 30, action size: 10
            >>> agent = DQN.mlp(tf.Session(), [30, 50, 50, 50, 10])
        """
        # Create target & evaluation network with multilayer perceptron
        e_net, e_input, e_vars = multilayer_perceptron(dimensions[:-1], alpha)
        t_net, t_input, t_vars = multilayer_perceptron(dimensions[:-1], alpha)
        return cls(
            session=session, eval_network=e_net, target_network=t_net,
            eval_input=e_input, target_input=t_input,
            eval_vars=e_vars, target_vars=t_vars,
            fc_dim=dimensions[-2], action_dim=dimensions[-1],
            learning_rate=learning_rate, discount=discount,
            replace_iter=replace_iter, alpha=alpha, logdir=logdir
        )

    @classmethod
    def cnn(cls, session, action_dim, channels, filters, poolings,
            width, height, depth=None, fc_dim=1024, learning_rate=1e-4,
            discount=0.9, replace_iter=100, alpha=1e-3, logdir=None):
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
            replace_iter (int): replace iteration (default: 100)
            alpha (float): slope of LeakyReLU (default: 1e-3)
        Returns:
            DQN
        Usage:
            >>> action_dim = 10 # Create 2D cnn
            >>> width, height = 1000, 500
            >>> channels = [3, 16, 32, 32] # input channel: 3 (RGB), output channel: 32
            >>> filters = [[5, 5], [5, 5], [3, 3]] # filter between channels
            >>> poolings = [[2, 2], [2, 2], [2, 2]] # pooling shape between channels
            >>> agent = DQN.cnn(
                    tf.Session(), action_dim, channels, filters, poolings,
                    width, height
                )
        """
        e_net, e_input, e_vars = conv_net(
            channels, filters, poolings, width, height, depth, fc_dim, alpha
        )
        t_net, t_input, t_vars = conv_net(
            channels, filters, poolings, width, height, depth, fc_dim, alpha
        )
        return cls(
            session=session, eval_network=e_net, target_network=t_net,
            eval_input=e_input, target_input=t_input,
            eval_vars=e_vars, target_vars=t_vars,
            fc_dim=fc_dim, action_dim=action_dim,
            learning_rate=learning_rate, discount=discount,
            replace_iter=replace_iter, alpha=alpha, logdir=logdir
        )

    def train(self, x, chosen_action, reward, next_x, done,
              next_available=None, learning_rate=None):
        if not learning_rate:
            learning_rate = self._learning_rate

        # Create available vector
        chosen_vector = self._create_action_vector(len(x), chosen_action)
        # Compute target value
        target_value = self._calculate_target_value(
            reward, next_x, done, next_available
        )

        if self._train_step % 100 == 0 and self._train_writer:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = self._session.run(
                [self._merged, self._train_op],
                feed_dict={
                    self._eval_input: x, self._chosen_holder: chosen_vector,
                    self._target_holder: target_value,
                    self._lr_input: learning_rate
                },
                options=run_options, run_metadata=run_metadata
            )
            self._train_writer.add_run_metadata(run_metadata, "step {}".format(self._train_step))
            self._train_writer.add_summary(summary, self._train_step)
        else:
            self._session.run(self._train_op, feed_dict={
                self._eval_input: x, self._chosen_holder: chosen_vector,
                self._target_holder: target_value,
                self._lr_input: learning_rate
            })

        # Update target network from evaluation network
        self._train_step += 1
        if self._train_step % self._replace_iter == 0:
            for r_op in self._replace_ops:
                self._session.run(r_op)

        return self.get_loss(
            x, chosen_action, reward, next_x, done, next_available
        )

    def get_loss(self, x, chosen_action, reward, next_x,
                 done, next_available=None):
        # Create available vector
        chosen_vector = self._create_action_vector(len(x), chosen_action)
        # Compute target value
        target_value = self._calculate_target_value(
            reward, next_x, done, next_available
        )

        return self._session.run(self._td_error, feed_dict={
            self._eval_input: x, self._chosen_holder: chosen_vector,
            self._target_holder: target_value,
        })

    def get_value(self, x):
        return self._session.run(
            self._eval_q, feed_dict={self._eval_input: x}
        )

    def _build_q(self):
        w_name, b_name = "w_out", "b_out"
        with tf.variable_scope("eval_network"):
            w = weight_variable([self._fc_dim, self._action_dim], w_name)
            b = bias_variable([self._action_dim], b_name)

            summarize_variable(w, w_name)
            summarize_variable(b, b_name)

            norm_w = normalize_weight(w, 0, name="norm_out")

            self._eval_vars[w_name] = w
            self._eval_vars[b_name] = b
            with tf.name_scope("eval_q") as scope:
                self._eval_q = tf.add(
                    tf.matmul(self._eval_network, norm_w), b,
                    name="eval_q"
                )

        with tf.variable_scope("target_network"):
            w = weight_variable([self._fc_dim, self._action_dim], w_name)
            b = bias_variable([self._action_dim], b_name)

            summarize_variable(w, w_name)
            summarize_variable(b, b_name)

            norm_w = normalize_weight(w, 0, name="norm_out")

            self._target_vars[w_name] = w
            self._target_vars[b_name] = b
            with tf.name_scope("target_q") as scope:
                self._target_q = tf.add(
                    tf.matmul(self._target_network, norm_w), b,
                    name="target_q"
                )

    def _build_tderror(self):
        with tf.variable_scope("DQN"):
            # Create holder for chosen
            self._chosen_holder = tf.placeholder(
                tf.float32, [None, self._action_dim], name="chosen_action"
            )
            self._target_holder = tf.placeholder(
                tf.float32, [None, 1], name="target_value"
            )

            # Create evaluation value
            self._eval_predict = tf.reduce_sum(
                tf.multiply(self._eval_q, self._chosen_holder),
                axis=1, keep_dims=True, name="eval_predict"
            )
            # Create TD error
            self._td_error = tf.subtract(
                self._target_holder, self._eval_predict, name="td_error"
            )

    def _build_train_op(self):
        # Opimize loss
        with tf.variable_scope("DQN"):
            self._lr_input = tf.placeholder(tf.float32, shape=[])
            self._loss = tf.losses.huber_loss(
                self._target_holder, self._eval_predict
            )
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._lr_input
            )
            self._train_op = optimizer.minimize(self._loss, name="train")
            tf.summary.scalar("loss", self._loss)

    def _build_replace_op(self):
        self._replace_ops = []
        for key, var in self._eval_vars.items():
            self._replace_ops.append(self._target_vars[key].assign(var))

    def _create_action_vector(self, batch_size, next_available=None):
        if next_available:
            available_vector = numpy.zeros([batch_size, self._action_dim])
            for batch_index, row in enumerate(next_available):
                for column_index in row:
                    available_vector[batch_index][column_index] = 1
        else:
            available_vector = numpy.ones([batch_size, self._action_dim])
        return available_vector

    def _calculate_target_value(self, reward, next_x, done,
                                next_available=None):
        # Get target Q
        target_q = self._session.run(self._target_q, feed_dict={
            self._target_input: next_x
        })
        # Compute target value from next state
        target_value = []
        for i in range(len(reward)):
            max_target = -1e9
            if next_available:
                availabel_list = next_available[i]
            else:
                availabel_list = range(self._action_dim)
            for a_index in availabel_list:
                if target_q[i][a_index] > max_target:
                    max_target = target_q[i][a_index]
            if done[i][0]:
                target_value.append([reward[i][0]])
            else:
                target_value.append([reward[i][0] + self._discount * max_target])
        return target_value


class DoubleDQN(DQN):

    def _calculate_target_value(self, reward, next_x, done,
                                next_available=None):
        # Use evaluation to select next state's action
        next_q = self._session.run(
            self._eval_q, feed_dict={self._eval_input: next_x}
        )
        target_q = self._session.run(
            self._target_q, feed_dict={self._target_input: next_x}
        )
        # Compute target value from next state
        target_value = []
        for i in range(len(reward)):
            max_target = -1e9
            max_index = 0
            if next_available:
                availabel_list = next_available[i]
            else:
                availabel_list = range(self._action_dim)
            for a_index in availabel_list:
                if next_q[i][a_index] > max_target:
                    max_target = next_q[i][a_index]
                    max_index = a_index
            if done[i][0]:
                target_value.append([reward[i][0]])
            else:
                target_value.append(
                    [reward[i][0] + self._discount * target_q[i][max_index]]
                )
        return target_value


class DuelingDQN(DQN):

    def get_value(self, x):
        return self._session.run(
            [self._eval_q, self._eval_adv], feed_dict={self._eval_input: x}
        )

    def _build_q(self):
        v_w_name, v_b_name = "val_w", "val_b"
        a_w_name, a_b_name = "adv_w", "adv_b"
        with tf.variable_scope("eval_network"):
            v_w = weight_variable([self._fc_dim, 1], v_w_name)
            v_b = bias_variable([1], v_b_name)
            a_w = weight_variable([self._fc_dim, self._action_dim], a_w_name)
            a_b = bias_variable([self._action_dim], a_b_name)
            self._eval_vars[v_w_name] = v_w
            self._eval_vars[v_b_name] = v_b
            self._eval_vars[a_w_name] = a_w
            self._eval_vars[a_b_name] = a_b

            summarize_variable(v_w, v_w_name)
            summarize_variable(v_b, v_b_name)
            summarize_variable(a_w, a_w_name)
            summarize_variable(a_b, a_b_name)

            norm_v_w = normalize_weight(v_w, 0, name="norm_val")
            norm_a_w = normalize_weight(a_w, 0, name="norm_adv")

            # Action-independent value function
            with tf.name_scope("eval_value") as scope:
                self._eval_val = tf.add(
                    tf.matmul(self._eval_network, norm_v_w), v_b,
                    name="eval_value")
            # Action-dependent advantage function
            with tf.name_scope("eval_advantage") as scope:
                self._eval_adv = tf.add(
                    tf.matmul(self._eval_network, norm_a_w), a_b,
                    name="eval_advantage"
                )
            # Q funciton
            with tf.name_scope("eval_q") as scope:
                self._eval_q = tf.add(
                    self._eval_val, self._eval_adv, name="eval_q"
                )

        with tf.variable_scope("target_network"):
            v_w = weight_variable([self._fc_dim, 1], v_w_name)
            v_b = bias_variable([1], v_b_name)
            a_w = weight_variable([self._fc_dim, self._action_dim], a_w_name)
            a_b = bias_variable([self._action_dim], a_b_name)
            self._target_vars[v_w_name] = v_w
            self._target_vars[v_b_name] = v_b
            self._target_vars[a_w_name] = a_w
            self._target_vars[a_b_name] = a_b

            summarize_variable(v_w, v_w_name)
            summarize_variable(v_b, v_b_name)
            summarize_variable(a_w, a_w_name)
            summarize_variable(a_b, a_b_name)

            norm_v_w = normalize_weight(v_w, 0, name="norm_val")
            norm_a_w = normalize_weight(a_w, 0, name="norm_adv")

            # Action-independent value function
            with tf.name_scope("target_value") as scope:
                self._target_val = tf.add(
                    tf.matmul(self._target_network, norm_v_w), v_b,
                    name="target_value"
                )
            # Action-dependent advantage function
            with tf.name_scope("target_advantage") as scope:
                self._target_adv = tf.add(
                    tf.matmul(self._target_network, norm_a_w), a_b,
                    name="target_advantage"
                )
            # Q funciton
            with tf.name_scope("target_q") as scope:
                self._target_q = tf.add(
                    self._target_val, self._target_adv, name="target_q"
                )

class DuelingDDQN(DuelingDQN, DoubleDQN):
    pass
