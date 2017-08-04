#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf


def summarize_variable(var, name):
    with tf.name_scope("summaries_{}".format(name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)
