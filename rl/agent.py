#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import six


@six.add_metaclass(ABCMeta)
class RLInterface():

    def __init__(self, **kwargs):
        """
        """
        self._session = kwargs['session']
        self._graph = self._session.graph

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, new_session):
        self._session = new_session

    @property
    def graph(self):
        return self._graph

    @abstractmethod
    def train(self, x, chosen_action, reward, next_x, next_available):
        pass

    @abstractmethod
    def get_loss(self, x, chosen_action, reward, next_x, next_available):
        pass

    @abstractmethod
    def get_value(self, state):
        pass
