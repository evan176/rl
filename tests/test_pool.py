#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import copy
from types import GeneratorType
from unittest import TestCase

try:
    from unittest import mock
except:
    import mock

from rl.pool import MemoryPool


class MemoryPoolTest(TestCase):
    
    def setUp(self):
        self.pool = MemoryPool(5000)
        self.test_data = [
            {
                'state': 1, 'action': 3, 'reward': 100,
                'next_state': 6, 'priority': 1
            },
            {
                'state': 2, 'action': 3, 'reward': 0,
                'next_state': 7, 'priority': 2
            },
            {
                'state': 3, 'action': 3, 'reward': -100,
                'next_state': 8, 'priority': 3
            },
            {
                'state': 4, 'action': 3, 'reward': 0,
                'next_state': 9, 'priority': 4
            },
            {
                'state': 5, 'action': 3, 'reward': 100,
                'next_state': 10, 'priority': 5
            },
        ]

        self.pool._experiences = copy(self.test_data)

    def tearDown(self):
        self.pool = None

    @mock.patch('rl.pool.MemoryPool.capacity')
    def test_add(self, mock_capacity):
        # Handle mock capacity
        mock_capacity.side_effect = lambda *args: len(self.pool._experiences)

        return_value = self.pool.add(6, '456', -100, None, [1213, 'a'], 1)
        self.assertEqual(return_value, 6)

    def test_add_negative(self):
        self.pool.add(6, '456', -100, None, [1213, 'a'], True, -1)
        self.assertEqual(self.pool._experiences[-1]['priority'], 1e-3)

    @mock.patch('rl.pool.MemoryPool.capacity')
    def test_remove(self, mock_capacity):
        # Handle mock capacity
        mock_capacity.return_value = len(self.pool._experiences)

        return_value = self.pool.remove(4)
        self.assertEqual(self.pool._experiences, self.test_data[:-1])

    def test_sample(self):
        returns = self.pool.sample(5)
        returns = sorted(returns, key=lambda x: x[0])
        expected = [(i, item) for i, item in enumerate(self.test_data)]
        self.assertEqual(returns, expected)

    def test_sample_greater(self):
        returns = self.pool.sample(10)
        returns = sorted(returns, key=lambda x: x[0])
        expected = [(i, item) for i, item in enumerate(self.test_data)]
        self.assertEqual(returns, expected)

    def test_update(self):
        self.pool.update([(0, 5), (1, 4), (2, 3), (3, 2), (4, 1)])
        for record, value in zip(self.pool._experiences, range(5, 0, -1)):
            self.assertEqual(record['priority'], value)

    def test_update_negative(self):
        self.pool.update([(0, 1), (1, -1), (2, 0), (3, 1), (4, 0)])
        self.assertEqual(self.pool._experiences[1]['priority'], 1e-3)

    def test_size(self):
        self.assertEqual(self.pool.size(), 5000)

    def test_capacity(self):
        self.assertEqual(self.pool.capacity(), 5)

    def test_all(self):
        self.assertIsInstance(self.pool.all(), GeneratorType)
        self.assertEqual(list(self.pool.all()), self.test_data)
