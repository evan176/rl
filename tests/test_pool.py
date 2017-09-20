#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bson.binary import Binary
from copy import copy
import os
import pickle
from pymongo import MongoClient
import random
import shutil
from types import GeneratorType
from unittest import TestCase

try:
    range = xrange
except:
    pass

try:
    from unittest import mock
except:
    import mock

import numpy
import pymongo
import six

from rl.pool import (
    filter_priority, bias_sample, encode_data, decode_data, save_data, load_data,
    MemoryPool, MongoPool, FilePool
)


class FilterPriorityTest(TestCase):

    def test_normal(self):
        self.assertEqual(filter_priority(3.01), 3.01)

    def test_less_0_001(self):
        self.assertEqual(filter_priority(1e-5), 1e-3)

    def test_greater_1000(self):
        self.assertEqual(filter_priority(1e+5), 1e+3)

    def test_negative(self):
        self.assertEqual(filter_priority(-1), 1e-3)

    def test_nan(self):
        self.assertEqual(filter_priority(numpy.nan), 1e-3)


class BiasSampleTest(TestCase):

    def test_size(self):
        pass

    def test_uniform(self):
        pass

    def test_bias(self):
        pass


class MemoryPoolInitTest(TestCase):

    def test_size(self):
        pool = MemoryPool(10)
        self.assertEqual(pool._size, 10)

    def test_size_negative(self):
        with self.assertRaises(TypeError):
            MemoryPool(-1)

    def test_size_non_integer(self):
        with self.assertRaises(TypeError):
            MemoryPool(3.15)

        with self.assertRaises(TypeError):
            MemoryPool("3.9")

    def test_var(self):
        pool = MemoryPool(10)
        self.assertEqual(pool._experiences, {})
        self.assertEqual(pool._q_front, 0)


class MemoryPoolTest(TestCase):
    
    def setUp(self):
        self._size = 50
        self._test_data = {
            0: {
                'state': 1, 'action': 3, 'reward': 100,
                'next_state': 6, 'priority': 1
            },
            1: {
                'state': 2, 'action': 3, 'reward': 0,
                'next_state': 7, 'priority': 2
            },
            2: {
                'state': 3, 'action': 3, 'reward': -100,
                'next_state': 8, 'priority': 3
            },
            3: {
                'state': 4, 'action': 3, 'reward': 0,
                'next_state': 9, 'priority': 4
            },
            4: {
                'state': 5, 'action': 3, 'reward': 100,
                'next_state': 10, 'priority': 5
            },
        }
        self._experiences = copy(self._test_data)

        self._pool = MemoryPool(self._size)
        self._pool._experiences = self._experiences
        self._pool._q_front = 4

    def tearDown(self):
        self._pool = None

    def _iter_data(self):
        for i in range(self._pool._q_front, len(self._experiences)):
            if i in self._experiences:
                yield i, self._experiences[i]
        for i in range(0, self._pool._q_front):
            if i in self._experiences:
                yield i, self._experiences[i]

    def _find_record(self, key):
        return self._experiences[key]

    def _get_first(self):
        if self._pool._q_front < self._size - 1:
            return self._experiences[self._pool._q_front + 1]
        else:
            return self._experiences[0]

    def _get_last(self):
        return self._experiences[self._pool._q_front]

    def _process(self, record):
        return record

    def test_add_argument(self):
        with self.assertRaises(TypeError):
            self._pool.add(1)

        with self.assertRaises(TypeError):
            self._pool.add(1, 2)

        with self.assertRaises(TypeError):
            self._pool.add(1, 2, 3)

        with self.assertRaises(TypeError):
            self._pool.add(1, 2, 3, 4)

    def test_add_data(self):
        new_data = {
            'state': 6, 'action': '456', 'reward': -100,
            'next_state': None, 'next_actions': [1213, 'a'],
            'done': True, 'priority': 1, 'info': 123
        }
        self._pool.add(**copy(new_data))
        for key, record in self._iter_data():
            record = self._process(record)
            print(record)
            if key in self._test_data:
                self.assertDictEqual(record, self._test_data[key])
            else:
                self.assertDictEqual(record, new_data)

    def test_add_negative_priority(self):
        new_data = {
            'state': 6, 'action': '456', 'reward': -100,
            'next_state': None, 'next_actions': [1213, 'a'],
            'done': True, 'priority': -1, 'info': 123
        }
        self._pool.add(**copy(new_data))
        self.assertEqual(
            self._process(self._get_last())['priority'], 1e-3
        )

    def test_add_large_priority(self):
        new_data = {
            'state': 6, 'action': '456', 'reward': -100,
            'next_state': None, 'next_actions': [1213, 'a'],
            'done': True, 'priority': 10000, 'info': 123
        }
        self._pool.add(**copy(new_data))
        self.assertEqual(
            self._process(self._get_last())['priority'], 1e+3
        )

    def test_add_greater_size(self):
        new_data = {
            'state': 6, 'action': '456', 'reward': -100,
            'next_state': None, 'next_actions': [1213, 'a'],
            'done': True, 'priority': 1, 'info': 123
        }
        len_test = len(self._test_data)
        for i in range(self._size - len_test + 1):
            self._pool.add(**copy(new_data))

        pool_data = [v for k, v in self._iter_data()]

        for k, record in self._iter_data():
            record = self._process(record)
            if k in range(1, len(self._test_data)):
                self.assertDictEqual(record, self._test_data[k])
            else:
                self.assertDictEqual(record, new_data)

    def test_add_size_zero(self):
        self._pool._size = 0
        new_data = {
            'state': 6, 'action': '456', 'reward': -100,
            'next_state': None, 'next_actions': [1213, 'a'],
            'done': True, 'priority': 1, 'info': 123
        }
        len_test = len(self._test_data)
        for i in range(50):
            self._pool.add(**copy(new_data))
        self.assertEqual(self._pool.amount(), len(self._test_data) + 50)

    def test_remove(self):
        for key, record in self._test_data.items():
            self._pool.remove(key)
            self.assertNotIn(record, [v for k, v in self._iter_data()])

    @mock.patch("rl.pool.bias_sample")
    def test_sample_type(self, mock_bias):
        mock_bias.side_effect = lambda x, y: list(range(y))

        num = int(len(self._test_data) / 2)
        self.assertIsInstance(self._pool.sample(num), GeneratorType)

    @mock.patch("rl.pool.bias_sample")
    def test_sample_data(self, mock_bias):
        mock_bias.side_effect = lambda x, y: list(range(y))

        num = int(len(self._test_data) / 2)
        data = []
        for key, record in self._pool.sample(num):
            record = self._process(record)
            self.assertIn(record, self._test_data.values())
            self.assertNotIn(record, data)
            data.append(record)

    def test_update(self):
        data = []
        for key in self._test_data.keys():
            data.append((key, random.uniform(1e-3, 1e+3)))

        self._pool.update(data)
        for i in range(len(data)):
            self.assertEqual(
                self._find_record(data[i][0])['priority'], data[i][1]
            )

    def test_update_less_0_001(self):
        data = []
        for key in self._test_data.keys():
            data.append((key, random.uniform(-1, 1e+3)))
        self._pool.update(data)
        for i in range(len(data)):
            if data[i][1] < 1e-3:
                value = 1e-3
            else:
                value = data[i][1]
            self.assertEqual(
                self._find_record(data[i][0])['priority'], value
            )

    def test_update_greater_1000(self):
        data = []
        for key in self._test_data.keys():
            data.append((key, random.uniform(1e-3, 1e+5)))
        self._pool.update(data)
        for i in range(len(data)):
            if data[i][1] > 1e+3:
                value = 1e+3
            else:
                value = data[i][1]
            self.assertEqual(
                self._find_record(data[i][0])['priority'], value
            )

    def test_size(self):
        self.assertEqual(self._pool.size(), self._size)

    def test_amount(self):
        self.assertEqual(self._pool.amount(), len(self._test_data))

    def test_all_type(self):
        self.assertIsInstance(self._pool.all(), GeneratorType)

    def test_all_data(self):
        data = {}
        for key, record in self._pool.all():
            data[key] = self._process(record)
        self.assertDictEqual(data, self._test_data)


class MongoPoolInitTest(TestCase):

    def setUp(self):
        self._client = MongoClient('test_mongo', 27017)
        self._db = self._client['test']
        self._coll = self._db['test']

    def tearDown(self):
        self._client.drop_database('test')

    def test_size(self):
        pool = MongoPool(self._coll, 10)

        self.assertEqual(pool._size, 10)
        self.assertEqual(pool._collection, self._coll)

    def test_size_non_positive(self):
        with self.assertRaises(TypeError):
            MongoPool(self._coll, -1)

    def test_size_non_integer(self):
        with self.assertRaises(TypeError):
            MongoPool(self._coll, 3.15)

        with self.assertRaises(TypeError):
            MongoPool(self._coll, "3.19")


class MongoPoolTest(MemoryPoolTest):

    def setUp(self):
        self._client = MongoClient('test_mongo', 27017)
        self._db = self._client['test']
        self._coll = self._db['test']

        self._size = 50
        self._test_data = {
            0: {
                'state': 1, 'action': 3, 'reward': 100,
                'next_state': 6, 'next_actions': [1, 2],
                'done': True, 'priority': 1, 'info': '123'
            },
            1: {
                'state': 2, 'action': 3, 'reward': 0,
                'next_state': 7, 'next_actions': [1, 2, [3, 4]],
                'done': False, 'priority': 2, 'info': [[34], [4]]
            },
            2: {
                'state': 3, 'action': 3, 'reward': -100,
                'next_state': 8, 'next_actions': [],
                'done': False, 'priority': 3, 'info': 10
            },
            3: {
                'state': 4, 'action': 3, 'reward': 0,
                'next_state': 9, 'next_actions': [-1, 10, 'eee'],
                'done': True, 'priority': 4, 'info': 3.4
            },
            4: {
                'state': 5, 'action': 3, 'reward': 100,
                'next_state': 10, 'next_actions': 3.0,
                'done': True, 'priority': 5, 'info': [23, 4]
            },
        }
        for key, item in six.iteritems(self._test_data):
            item = encode_data(copy(item))
            item['index'] = key
            self._coll.insert_one(item)

        self._pool = MongoPool(self._coll, self._size)

    def tearDown(self):
        self._client.drop_database('test')

    def _iter_data(self):
        for record in self._coll.find().sort("index", 1):
            yield record['index'], record

    def _find_record(self, key):
        return self._coll.find_one({'index': key})

    def _get_first(self):
        return self._coll.find().sort("index", 1).limit(1)[0]

    def _get_last(self):
        return self._coll.find().sort("index", -1).limit(1)[0]

    def _process(self, record):
        record = copy(record)
        del record['_id']
        del record['index']
        try:
            return decode_data(record)
        except:
            return record


class FilePoolInitTest(TestCase):

    def setUp(self):
        self._dir = "test_pool"

    def tearDown(self):
        if os.path.isdir(self._dir):
            shutil.rmtree(self._dir)

    def test_size(self):
        pool = FilePool(self._dir, 10)

        self.assertEqual(pool._size, 10)

    def test_size_non_positive(self):
        with self.assertRaises(TypeError):
            FilePool(self._dir, -1)

    def test_size_non_integer(self):
        with self.assertRaises(TypeError):
            FilePool(self._dir, 3.15)

        with self.assertRaises(TypeError):
            FilePool(self._dir, "3.19")

    def test_folder(self):
        FilePool(self._dir, 10)

        self.assertTrue(os.path.isdir(self._dir))


class FilePoolTest(MemoryPoolTest):

    def setUp(self):
        self._dir = "test_pool"
        self._size = 50

        self._test_data = {
            "test_pool/0.npz": {
                'state': [1], 'action': [3], 'reward': [100],
                'next_state': [6], 'next_actions': [1, 2],
                'done': True, 'priority': 1, 'info': '123'
            },
            "test_pool/1.npz": {
                'state': [2], 'action': [3], 'reward': 0,
                'next_state': 7, 'next_actions': [[1, 2], [3, 4]],
                'done': False, 'priority': 2, 'info': [[34], [4]]
            },
            "test_pool/2.npz": {
                'state': 3, 'action': 3, 'reward': -100,
                'next_state': 8, 'next_actions': [],
                'done': False, 'priority': 3, 'info': 10
            },
            "test_pool/3.npz": {
                'state': 4, 'action': 3, 'reward': 0,
                'next_state': 9, 'next_actions': [-1, 10],
                'done': True, 'priority': 4, 'info': 3.4
            },
            "test_pool/4.npz": {
                'state': 5, 'action': 3, 'reward': 100,
                'next_state': 10, 'next_actions': 3.0,
                'done': True, 'priority': 5, 'info': [23, 4]
            },
        }

        self._pool = FilePool(self._dir, self._size)
        for key, record in six.iteritems(self._test_data):
            path = save_data(
                key.split(".npz")[0], record['state'],
                record['action'], record['reward'], record['next_state'],
                record['done'], record['next_actions'], record['priority'],
                record['info']
            )
            self._pool._indexes[path] = 1

    def tearDown(self):
        if os.path.isdir(self._dir):
            shutil.rmtree(self._dir)

    def _iter_data(self):
        indexes = {}
        for dir_path, dir_names, file_names in os.walk(self._dir):
            for file_name in file_names:
                if file_name[-4:] == ".npz":
                    key = int(file_name.split('.npz')[0])
                    indexes[key] = os.path.join(dir_path, file_name)
        for key in sorted(six.iterkeys(indexes)):
            yield indexes[key], load_data(indexes[key])

    def _find_record(self, key):
        record = copy(load_data(key))
        record['priority'] = self._pool._indexes[key]
        return record

    def _get_first(self):
        indexes = {}
        for dir_path, dir_names, file_names in os.walk(self._dir):
            for file_name in file_names:
                if file_name[-4:] == ".npz":
                    key = int(file_name.split('.npz')[0])
                    indexes[key] = os.path.join(dir_path, file_name)
        key = min(six.iterkeys(indexes))
        return load_data(indexes[key])

    def _get_last(self):
        indexes = {}
        for dir_path, dir_names, file_names in os.walk(self._dir):
            for file_name in file_names:
                if file_name[-4:] == ".npz":
                    key = int(file_name.split('.npz')[0])
                    indexes[key] = os.path.join(dir_path, file_name)
        key = max(six.iterkeys(indexes))
        return load_data(indexes[key])

    def _process(self, record):
        return record
