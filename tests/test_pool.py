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
        self.test_data = {
            0: {
                'state': 1, 'action': 3, 'reward': 100,
                'next_state': 6, 'priority': 1
            },
            1: {
                'state': 2, 'action': 3, 'reward': 0,
                'next_state': 7, 'priority': 2
            },
            -1: {
                'state': 3, 'action': 3, 'reward': -100,
                'next_state': 8, 'priority': 3
            },
            2: {
                'state': 4, 'action': 3, 'reward': 0,
                'next_state': 9, 'priority': 4
            },
            4: {
                'state': 5, 'action': 3, 'reward': 100,
                'next_state': 10, 'priority': 5
            },
        }
        self.pool._experiences = copy(self.test_data)

    def tearDown(self):
        self.pool = None

    @mock.patch('rl.pool.MemoryPool.capacity')
    def test_add(self, mock_capacity):
        # Handle mock capacity
        mock_capacity.side_effect = lambda *args: len(self.pool._experiences)

        return_value = self.pool.add(6, '456', -100, None, [1213, 'a'], 1)
        self.assertEqual(self.pool._q_front, 3)
        self.assertEqual(return_value, 6)

    def test_add_negative(self):
        self.pool.add(6, '456', -100, None, [1213, 'a'], True, -1)
        self.assertEqual(self.pool._q_front, 3)
        self.assertEqual(
            self.pool._experiences[self.pool._q_front]['priority'], 1e-3
        )

    @mock.patch('rl.pool.MemoryPool.capacity')
    def test_remove(self, mock_capacity):
        # Handle mock capacity
        mock_capacity.return_value = len(self.pool._experiences)

        return_value = self.pool.remove(4)
        self.assertNotIn(4, self.pool._experiences)
        self.assertEqual(return_value, self.test_data[4])

    def test_sample(self):
        for key, record in self.pool.sample(5):
            self.assertEqual(record, self.test_data[key])

    def test_sample_greater(self):
        for key, record in self.pool.sample(10):
            self.assertEqual(record, self.test_data[key])

    def test_update(self):
        data = [(0, 5), (1, 4), (-1, 3), (2, 2), (4, 1)]
        self.pool.update(data)
        for i in range(len(data)):
            self.assertEqual(self.pool._experiences[data[i][0]]['priority'], data[i][1])

    def test_update_negative(self):
        self.pool.update([(0, 1), (1, -1), (2, 0), (3, 1), (4, 0)])
        self.assertEqual(self.pool._experiences[1]['priority'], 1e-3)

    def test_size(self):
        self.assertEqual(self.pool.size(), 5000)

    def test_capacity(self):
        self.assertEqual(self.pool.capacity(), 5)

    def test_all(self):
        self.assertIsInstance(self.pool.all(), GeneratorType)
        result_dict = {}
        for key, record in self.pool.all():
            result_dict[key] = record
        for key, record in self.test_data.items():
            self.assertEqual(record, self.test_data[key])
