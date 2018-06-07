#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
from bson.binary import Binary
import os
import pickle
import uuid

try:
    range = xrange
except:
    pass

import numpy
import pymongo
import six


@six.add_metaclass(ABCMeta)
class PoolInterface():
    """
    Pool interface defintion

    """
    @abstractmethod
    def add(self, state, action, reward, next_state, done,
            next_actions=None, priority=1, info=None):
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
    def remove(self, index):
        """
        Remove record from pool with index
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


def filter_priority(value):
    if numpy.isnan(value):
        return 1e-3
    elif value < 1e-3:
        return 1e-3
    elif value > 1e+3:
        return 1e+3
    return value


def bias_sample(dist, size):
    if size > 0 and dist:
        if size > len(dist):
            samples = range(len(dist))
        else:
            sum_d = float(sum(dist))
            prob = [item / sum_d for item in dist]

            samples = numpy.random.choice(
                range(len(dist)),
                size=size, p=prob, replace=False
            )
            samples = samples.tolist()
    else:
        samples = []
    return samples


def encode_data(data):
    data['state'] = Binary(pickle.dumps(data['state']))
    data['action'] = Binary(pickle.dumps(data['action']))
    data['reward'] = Binary(pickle.dumps(data['reward']))
    data['next_state'] = Binary(pickle.dumps(data['next_state']))
    data['next_actions'] = Binary(pickle.dumps(data['next_actions']))
    return data


def decode_data(data):
    data['state'] = pickle.loads(data['state'])
    data['action'] = pickle.loads(data['action'])
    data['reward'] = pickle.loads(data['reward'])
    data['next_state'] = pickle.loads(data['next_state'])
    data['next_actions'] = pickle.loads(data['next_actions'])
    return data


def save_data(file_name, state, action, reward, next_state,
              done, next_actions, priority, info):
    """
    Save numpy array to disk

    Args:
        state: any type as long as it can describe the state of environment
        action: any type as long as it can represent executed action with
            above state
        reward: also free type for action feedback
        next_state: Like state but it is for describing next state
        next_actions: For next state's actions (Default: None),

    Returns:
        path (str): file path

    Examples:
        >>> fpool._save_data(state, action, ...)
        "/xxx/yyy/zzz/f88fc90a-bc3d-413f-8864-44b21beb7bc5.npz"

    """
    numpy.savez(
        file_name, state=state, action=action, reward=reward,
        next_state=next_state, done=done, next_actions=next_actions,
        priority=priority, info=info
    )
    return "{}.npz".format(file_name)


def load_data(path):
    """
    Load data from disk to memory

    Args:
        path (str): file path


    Returns:
        record (dict): If IOError then return None

    Examples:

    """
    try:
        npzfile = numpy.load(path)
        return {
            "state": npzfile["state"],
            "action": npzfile["action"],
            "reward": npzfile["reward"],
            "next_state": npzfile["next_state"],
            "done": npzfile["done"],
            "next_actions": npzfile["next_actions"],
            "priority": npzfile["priority"],
            "info": npzfile["info"],
        }
    except IOError:
        return None


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
        if not isinstance(pool_size, int):
            raise TypeError("Pool size should be positive integer")
        if pool_size < 0:
            raise TypeError("Pool size should be positive integer")

        self._size = pool_size
        self._experiences = {}
        self._q_front = 0

        self._tree = numpy.zeros([2 * pool_size - 1])

    def add(self, state, action, reward, next_state, done,
            next_actions=None, priority=1, info=None):
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
        priority = filter_priority(priority)
        if self._q_front + 1 >= self._size > 0:
            self._q_front = 0
        else:
            self._q_front += 1

        self._update_tree(self._q_front + self._size - 1, priority)
        # Add new data
        self._experiences[self._q_front] = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'next_actions': next_actions,
            'done': done,
            'priority': priority,
            'info': info,
        }

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self._tree[parent] += change

        while parent != 0:
            parent = (parent - 1) // 2
            self._tree[parent] += change

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        while left < len(self._tree):
            if s <= self._tree[left]:
                idx = left
            else:
                s = s - self._tree[left]
                idx = right
            left = 2 * idx + 1
            right = left + 1
        return idx

    def _update_tree(self, idx, p):
        change = p - self._tree[idx]

        self._tree[idx] = p
        self._propagate(idx, change)

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
        del self._experiences[key]

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
                (...)
                ]

        Examples:
            # Biased random sampling 100 records
            >>> pool.sample(100)

        """
        """
        idx_memory = []
        while len(idx_memory) < size:
            new_idx = self._retrieve(0, random.uniform(1e-3, self._tree[0])) - self._size + 1
            if sum([abs(new_idx - idx) > size for idx in idx_memory]) == len(idx_memory):
                idx_memory.append(new_idx)
                yield new_idx, self._experiences[new_idx]
        """
        for i in range(size):
            new_idx = self._retrieve(0, random.uniform(1e-3, self._tree[0])) - self._size + 1
            yield new_idx, self._experiences[new_idx]

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
            try:
                self._update_tree(key + self._size - 1, filter_priority(priority))
            except (KeyError, TypeError) as e:
                print(e)

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
        for key, record in six.iteritems(self._experiences):
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
        if not isinstance(pool_size, int):
            raise TypeError("Pool size should be positive integer")
        if pool_size < 0:
            raise TypeError("Pool size should be positive integer")
        if not isinstance(collection, pymongo.collection.Collection):
            raise TypeError("Collection should be pymongo.Collection")

        self._size = pool_size
        self._collection = collection

    def add(self, state, action, reward, next_state, done,
            next_actions=None, priority=1, info=None):
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
        priority = filter_priority(priority)

        last_record = self._get_last()
        if last_record:
            index = last_record['index'] + 1
            if self.amount() >= self.size() > 0:
                first_record = self._get_first()
                self.remove(first_record["index"])
        else:
            index = 0

        self._collection.insert_one(encode_data({
            'index': index, 'state': state, 'action': action,
            'reward': reward, 'next_state': next_state,
            'next_actions': next_actions, 'done': done,
            'priority': priority, 'info': info,
        }))

    def remove(self, index):
        """
        Remove record from pool with index.

        Args:
            index: the index of data

        Returns:
            None

        Examples:
            # Remove 100th data in pool
            >>> pool.remove(100)

        """
        self._collection.remove({'index': index})

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
            print(record)
            dist.append(record['priority'])
            indexes.append(record['index'])
        
        samples = bias_sample(dist, size)

        condition = {'index': {'$in': [indexes[i] for i in samples]}}
        for record in self._collection.find(condition):
            yield record['index'], decode_data(record)

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
            self._collection.update_one(
                {"index": index},
                {"$set": {"priority": filter_priority(priority)}}
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
        for record in self._collection.find().sort("index", 1):
            yield record['index'], decode_data(record)

    def _get_first(self):
        if self.amount() > 0:
            return self._collection.find().sort("index", 1).limit(1)[0]
        else:
            return None

    def _get_last(self):
        if self.amount() > 0:
            return self._collection.find().sort("index", -1).limit(1)[0]
        else:
            return None


class FilePool(PoolInterface):
    """
    FilePool store experience data to file system. Data must be numpy array.

    Args:
        directory (str): directory for storing data
        pool_size (int): sepcify size of pool. `0` for unlimited
            (default: 0)

    Returns:
        FilePool object

    Examples:
        # Init pool
        >>> fpool = FilePool("experiences")
        # Add data to pool
        >>> fpool.add(
                state=[1, 2, 3], action=3, reward=100, next_state=[4, 5, 6]
            )

        # Sample data for training
        >>> for path, record in fpool.sample(30):
        >>>     ...

        # Update priority of data
        >>> priorities = [(path, 10), (path, 0), (path, 9), ...]
        >>> fpool.update(priorities)
    """

    def __init__(self, directory, pool_size=0):
        if not isinstance(pool_size, int):
            raise TypeError("Pool size should be positive integer")
        if pool_size < 0:
            raise TypeError("Pool size should be positive integer")

        self._size = pool_size
        self._dir = directory
        self._make_dirs(self._dir)
        self._indexes = dict()
        # Synchronization
        self.sync()

    def add(self, state, action, reward, next_state, done,
            next_actions=None, priority=1, info=None):
        """
        Add new data to pool.

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
            >>> fpool.add(
                    state=[0, 0, 1], action=1,
                    reward=100, next_state=[1, 0, 0]
                )
            >>> fpool.add(
                    state={'a': 1, 'b': 0}, action=3,
                    reward=-1, next_state={'a': -1, 'b': 1},
                    next_actions=[3, 1, 0], priority=3.5
                )

        """
        priority = filter_priority(priority)
        if self.amount() + 1 >= self._size > 0:
            path = self._get_first()
            if path:
                self.remove(path)

        path = self._get_last()
        if path:
            file_name = str(int(os.path.basename(path).split('.npz')[0]) + 1)
        else:
            file_name = "0.npz"

        path = save_data(
            os.path.join(self._dir, file_name),
            state, action, reward, next_state, done, next_actions,
            priority, info
        )
        self._indexes[path] = priority

    def remove(self, path):
        """
        Remove record from pool with path.

        Args:
            path: the path of specific data

        Returns:
            None

        Examples:
            # Remove in pool with key: "~/experiences/sample.npz"
            >>> pool.remove("~/experiences/smaple.npz")

        """
        self._remove_data(path)
        del self._indexes[path]

    def sample(self, size):
        """
        Sample records from pool with given size.

        Args:
            size (int): sampling size

        Returns:
            path, record (generator):
                index,
                {
                    'state': ..., 'action': ..., 'reward': ...,
                    'next_state': ..., 'next_actions': ..., 'priority': ...,
                }

        Examples:
            # Biased random sampling 100 records
            >>> pool.sample(100)

        """
        dist = []
        paths = []
        for path, p in six.iteritems(self._indexes):
            dist.append(p)
            paths.append(path)

        samples = bias_sample(dist, size)

        for path in [paths[i] for i in samples]:
            record = load_data(path)
            record["priority"] = self._indexes[path]
            yield path, record

    def update(self, priorities):
        """
        Update each records' priority

        Args:
            priorities (list):
                [
                 (path, priority),
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
        for path, priority in priorities:
            if path in self._indexes:
                if path in self._indexes:
                    self._indexes[path] = filter_priority(priority)

    def size(self):
        """
        Get size of current pool

        Args:
            None

        Returns:
            pool_size (int): limited size of pool

        Examples:
            >>> fpool = FilePool(300)
            >>> print(fpool.size())
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
            >>> fpool = FilePool(300)
            >>> print(fpool.size())
            300
            >>> print(fpool.amount())
            0
            >>> fpool.add(1, 2, 3, 4)
            >>> print(fpool.amount())
            1

        """
        return len(self._indexes)

    def all(self):
        """
        Get all experiences of pool by generator

        Args:
            None

        Returns:
            path, record (generator): experience record

        Examples:
            >>> fpool = FilePool(300)
            >>> for path, item in fpool.all():
                    print(item)
            {'state': ...}
            {'state': ...}
            ...

        """
        for path, priority in six.iteritems(self._indexes):
            record = load_data(path)
            if record:
                record['priority'] = priority
                yield path, record

    def sync(self):
        """
        Synchronize indexes from file system

        Args:
            None

        Returns:
            None

        Examples:

        """
        for dir_path, dir_names, file_names in os.walk(self._dir):
            for file_name in file_names:
                if file_name[-4:] == ".npz":
                    path = os.path.join(dir_path, file_name)
                    if path not in self._indexes:
                        self._indexes[path] = 1

    def _make_dirs(self, path):
        """
        Create folder if it is not exists

        Args:
            path (str): target folder

        Returns:
            None

        Examples:

        """
        if not os.path.exists(path):
            os.makedirs(path)



    def _remove_data(self, path):
        """
        Remove data from disk

        Args:
            path (str): file path

        Returns:
            None

        Examples:

        """
        try:
            os.remove(path)
        except IOError:
            pass

    def _get_first(self):
        self.sync()
        if self.amount() > 0:
            paths = list(six.iterkeys(self._indexes))
            paths = sorted(paths, key=lambda x: int(os.path.basename(x).split(".npz")[0]))
            return paths[0]
        else:
            return None

    def _get_last(self):
        if self.amount() > 0:
            paths = list(six.iterkeys(self._indexes))
            paths = sorted(paths, key=lambda x: int(os.path.basename(x).split(".npz")[0]))
            return paths[-1]
        else:
            return None
