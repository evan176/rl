#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from bson.binary import Binary
import math
import pickle

import numpy
import six


@six.add_metaclass(ABCMeta)
class PoolInterface():
    """
    Pool interface defintion

    """
    @abstractmethod
    def add(self, state, action, reward, next_state, done,
            next_actions=None, priority=1):
        """
        Pool is used to store experience data, like:
          (state, action, reward, next_state). It must contains 4 things :
            state: state of environment
            action: executed action with given state
            reward: feedback of action with given state
            next_state: next state of environment after executing action
        These 2 things is optional:
            next_actions: available actions of next state
            priority: priority (loss) of this record
        """
        pass

    @abstractmethod
    def remove(self, record_id):
        """
        Remove record from pool with record_id
        """
        pass

    @abstractmethod
    def sample(self, size):
        """
        Sample records from pool with given size.
        """
        pass

    @abstractmethod
    def update(self, priorities):
        """
        Update each records' priority
        """
        pass

    @abstractmethod
    def size(self):
        """
        Get size of current pool
        """
        pass

    @abstractmethod
    def amount(self):
        """
        Get number of records in pool
        """
        pass

    @abstractmethod
    def all(self):
        """
        Get all experiences of pool by generator
        """
        pass


class MemoryPool(PoolInterface):
    """
    MemoryPool uses `dict` to store experience data. Data can be sampled
    after it add to pool. Sample method is biased random sampling with
    priority.

    Args:
        pool_size (int): sepcify size of memory pool. `0` for unlimited
            (default: 0)

    Returns:
        MemoryPool object

    Examples:
        # Init pool
        >>> mpool = MemoryPool(3000)
        # Add data to pool
        >>> mpool.add(
                state=[1, 2, 3], action=3, reward=100, next_state=[4, 5, 6]
            )
        # Sample data for training
        >>> records = mpool.sample(30)
        >>> print(len(records))
        30
        # Update priority of data
        >>> priorities = [(key1, 10), (key2, 0), (key3, 9), ...]
        >>> mpool.update(priorities)
    """

    def __init__(self, pool_size=0):
        if isinstance(pool_size, int):
            if pool_size < 0:
                raise TypeError("Pool size should be positive integer")
            self._size = pool_size

        self._experiences = {}
        self._q_front = 0

    def add(self, state, action, reward, next_state, done,
            next_actions=None, priority=1):
        """
        Add new data to experience pool.

        Args:
            state: any type as long as it can describe the state of environment
            action: any type as long as it can represent executed action with
                above state
            reward: also free type for action feedback
            next_state: Like state but it is for describing next state
            next_actions: For next state's actions (Default: None),
            priority: It specify the priority of data (Default: 0)

        Returns:
            amount: record number in pool

        Examples:
            >>> mpool.add(
                    state=[0, 0, 1], action=1,
                    reward=100, next_state=[1, 0, 0]
                )
            >>> mpool.add(
                    state={'a': 1, 'b': 0}, action=3,
                    reward=-1, next_state={'a': -1, 'b': 1},
                    next_actions=[3, 1, 0], priority=3.5
                )

        """
        if numpy.isnan(priority):
            priority = 1e-3
        elif priority < 1e-3:
            priority = 1e-3
        elif priority > 1e+3:
            priority = 1e+3

        if self._q_front > six.MAXSIZE:
            self._q_front = 0

        while self._q_front in self._experiences:
            self._q_front += 1

        self._experiences[self._q_front] = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'next_actions': next_actions,
            'done': done,
            'priority': priority,
        }

        if self.amount() > self.size() > 0:
            min_p = 1e+9
            min_key = 0
            for key, record in self._experiences.items():
                if record['priority'] < min_p:
                    min_p = record['priority']
                    min_key = key
            self.remove(min_key)
        return self.amount()

    def remove(self, key):
        """
        Remove record from pool with key.

        Args:
            key: the key of record in dictionary

        Returns:
            None

        Examples:
            # Remove 100th data in pool
            >>> pool.remove(100)

        """
        return self._experiences.pop(key)

    def sample(self, size):
        """
        Sample records from pool with given size.

        Args:
            size (int): sampling size

        Returns:
            samples (list):
                [
                (index, {'state': ..., 'action': ..., 'reward': ...,
                 'next_state': ..., 'next_actions': ..., 'priority': ...,}),
                (...),
                ]

        Examples:
            # Biased random sampling 100 records
            >>> pool.sample(100)

        """
        dist = []
        keys = []
        for k, record in self._experiences.items():
            dist.append(record['priority'])
            keys.append(k)

        sum_d = float(sum(dist))
        prob = [item / sum_d for item in dist]

        if size > 0:
            if size > self.amount():
                size = self.amount()
            keys = numpy.random.choice(
                keys, size=size, p=prob, replace=False
            )
        else:
            keys = []

        for k in keys:
            yield k, self._experiences[k]

    def update(self, priorities):
        """
        Update each records' priority

        Args:
            priorities (list):
                [
                 (index, priority),
                 (...),
                 ...
                ]

        Returns:
            None

        Examples:
            >>> pool.update([
                    (0, 10),
                    (1, 0),
                    (2, 3)
                ])

        """
        for key, priority in priorities:
            if numpy.isnan(priority):
                p = 1e-3
            elif priority < 1e-3:
                p = 1e-3
            elif priority > 1e+3:
                p = 1e+3
            else:
                p = priority

            try:
                self._experiences[key]['priority'] = p
            except:
                pass

    def size(self):
        """
        Get size of current pool

        Args:
            None

        Returns:
            pool_size (int): limited size of pool

        Examples:
            >>> mpool = MemoryPool(300)
            >>> print(mpool.size())
            300

        """
        return self._size

    def amount(self):
        """
        Get number of records in pool

        Args:
            None

        Returns:
            number (int): number of records

        Examples:
            >>> mpool = MemoryPool(300)
            >>> print(mpool.size())
            300
            >>> print(mpool.amount())
            0
            >>> mpool.add(1, 2, 3, 4)
            >>> print(mpool.amount())
            1

        """
        return len(self._experiences)

    def all(self):
        """
        Get all experiences of pool by generator

        Args:
            None

        Returns:
            record (dict): experience record

        Examples:
            >>> mpool = MemoryPool(300)
            >>> for item in mpool.all():
                    print(item)
            {'state': ...}
            {'state': ...}
            ...

        """
        for key, record in self._experiences.items():
            yield key, record


class MongoPool(PoolInterface):
    """
    MongoPool store experience data to db. Data must be numpy array.

    Args:
        collection (pymongo.collection.Collection): Specific collection for
            storing experience data
        pool_size (int): sepcify size of pool. `0` for unlimited
            (default: 0)

    Returns:
        MongoPool object

    Examples:
        # Init pool
        >>> client = MongoClient()
        >>> mpool = MongoPool(client['DB']['Collection'])
        # Add data to pool
        >>> mpool.add(
                state=[1, 2, 3], action=3, reward=100, next_state=[4, 5, 6]
            )
        # Sample data for training
        >>> records = mpool.sample(30)
        >>> print(len(records))
        30
        # Update priority of data
        >>> priorities = [(id1, 10), (id2, 0), (id3, 9), ...]
        >>> mpool.update(priorities)
    """

    def __init__(self, collection, pool_size=0):
        if isinstance(pool_size, int):
            if pool_size < 0:
                raise TypeError("Pool size should be positive integer")
            self._size = pool_size

        self._collection = collection

    def add(self, state, action, reward, next_state, done,
            next_actions=None, priority=1):
        """
        Add new data to experience pool.

        Args:
            state: any type as long as it can describe the state of environment
            action: any type as long as it can represent executed action with
                above state
            reward: also free type for action feedback
            next_state: Like state but it is for describing next state
            next_actions: For next state's actions (Default: None),
            priority: It specify the priority of data (Default: 0)

        Returns:
            amount: record number in pool

        Examples:
            >>> mpool.add(
                    state=[0, 0, 1], action=1,
                    reward=100, next_state=[1, 0, 0]
                )
            >>> mpool.add(
                    state={'a': 1, 'b': 0}, action=3,
                    reward=-1, next_state={'a': -1, 'b': 1},
                    next_actions=[3, 1, 0], priority=3.5
                )

        """
        if numpy.isnan(priority):
            priority = 1e-3
        elif priority < 1e-3:
            priority = 1e-3
        elif priority > 1e+3:
            priority = 1e+3

        last_record = self._get_last()
        if last_record:
            index = last_record['index'] + 1
        else:
            index = 0

        data = {
            'index': index,
            'state': Binary(pickle.dumps(state)),
            'action': Binary(pickle.dumps(action)),
            'reward': Binary(pickle.dumps(reward)),
            'next_state': Binary(pickle.dumps(next_state)),
            'next_actions': Binary(pickle.dumps(next_actions)),
            'done': done,
            'priority': priority,
        }

        self._collection.insert_one(data)
        if self.amount() > self.size() > 0:
            min_p = 1e+9
            min_index = 0
            for i, record in enumerate(self._experiences):
                if record['priority'] < min_p:
                    min_p = record['priority']
                    min_index = i
            self.remove(min_index)
        return self.amount()

    def remove(self, record_id):
        """
        Remove record from pool with record_id.

        Args:
            record_id: the index of data

        Returns:
            None

        Examples:
            # Remove 100th data in pool
            >>> pool.remove(100)

        """
        self._collection.remove({'index': record_id})

    def sample(self, size):
        """
        Sample records from pool with given size.

        Args:
            size (int): sampling size

        Returns:
            samples (list):
                [
                (index, {'state': ..., 'action': ..., 'reward': ...,
                 'next_state': ..., 'next_actions': ..., 'priority': ...,}),
                (...),
                ]

        Examples:
            # Biased random sampling 100 records
            >>> pool.sample(100)

        """
        dist = []
        indexes = []
        for record in self._collection.find({}, {'index': 1, 'priority': 1}):
            dist.append(record['priority'])
            indexes.append(record['index'])

        sum_d = float(sum(dist))
        prob = [item / sum_d for item in dist]

        if size > 0 and indexes:
            if size > len(indexes):
                size = len(indexes)

            indexes = numpy.random.choice(
                indexes,
                size=size, p=prob, replace=False
            )
            indexes = indexes.tolist()
        else:
            indexes = []

        samples = list()
        for record in self._collection.find({'index': {'$in': indexes}}):
            record['state'] = pickle.loads(record['state'])
            record['action'] = pickle.loads(record['action'])
            record['reward'] = pickle.loads(record['reward'])
            record['next_state'] = pickle.loads(record['next_state'])
            record['next_actions'] = pickle.loads(record['next_actions'])
            samples.append((record['index'], record))
        return samples

    def update(self, priorities):
        """
        Update each records' priority

        Args:
            priorities (list):
                [
                 (index, priority),
                 (...),
                 ...
                ]

        Returns:
            None

        Examples:
            >>> pool.update([
                    (0, 10),
                    (1, 0),
                    (2, 3)
                ])

        """
        for index, priority in priorities:
            if numpy.isnan(priority):
                p = 1e-3
            elif priority < 1e-3:
                p = 1e-3
            elif priority > 1e+3:
                p = 1e+3
            else:
                p = priority

            self._collection.update_one(
                {"index": index}, {"$set": {"priority": p}}
            )

    def size(self):
        """
        Get size of current pool

        Args:
            None

        Returns:
            pool_size (int): limited size of pool

        Examples:
            >>> mpool = MemoryPool(300)
            >>> print(mpool.size())
            300

        """
        return self._size

    def amount(self):
        """
        Get number of records in pool

        Args:
            None

        Returns:
            number (int): number of records

        Examples:
            >>> mpool = MemoryPool(300)
            >>> print(mpool.size())
            300
            >>> print(mpool.amount())
            0
            >>> mpool.add(1, 2, 3, 4)
            >>> print(mpool.amount())
            1

        """
        return self._collection.count()

    def all(self):
        """
        Get all experiences of pool by generator

        Args:
            None

        Returns:
            record (dict): experience record

        Examples:
            >>> mpool = MemoryPool(300)
            >>> for item in mpool.all():
                    print(item)
            {'state': ...}
            {'state': ...}
            ...

        """
        condition = {"$query": {}, "$orderby": {"id": 1}}
        for record in self._collection.find(condition):
            record['state'] = pickle.loads(record['state'])
            record['action'] = pickle.loads(record['action'])
            record['reward'] = pickle.loads(record['reward'])
            record['next_state'] = pickle.loads(record['next_state'])
            record['next_actions'] = pickle.loads(record['next_actions'])
            yield record

    def _get_last(self):
        condition = {"$query": {}, "$orderby": {"index": -1}}
        return self._collection.find_one(condition)
