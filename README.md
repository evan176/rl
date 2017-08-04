# RL

This package is for reinforcement learning experiment. It includes some famous algorithms (DQN, ActorCritic). Priorized experience replay is also implemented.

## Getting Started
### Prerequisites
1. tensorflow == 1.2.1
2. numpy >= 1.8
3. pymongo == 3.4.0
### Installing
1. Git clone
```
git clone https://github.com/evan176/rl.git
```
2. You can use make to create virtual environment and install dependency packages
```
make build35 install
```
3. Directly install
```
python setup.py install
```
## Example
### Import tensorflow and DQN
```
>>> import tensorflow as tf
>>> from rl import DQN, MemoryPool
```
### Create DQN
Create a Deep Q Network with multi layer perceptron with the observation size 10 and 3 actions. The hidden layer size is 5.
```
>>> agent = DQN.mlp(tf.Session(), [10, 5, 3])
```
### Train
Generate 2 records to train (State, Action, Reward, NextState)
1. ([0, 1, 0, 1, 0], 2, 1, [0, 0.5, 0, 0, 0.5])
2. ([1, 0, 1, 0, 1], 0, 0, [0.5, 0, 0.5, 0, 0.5])

```
>>> agent.train(
>>>     x=[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
>>>     chosen_action=[[2], [0]],
>>>     reward=[[1], [0]],
>>>     next_x=[[0, 0.5, 0, 0, 0.5], [0.5, 0, 0.5, 0, 0.5]]
>>> )
```
### Get Q value
Get action value of state
```
>>> print(agent.get_value([[1, 1, 0, 1, 1]]))
[1.33, -0.9, 2.46]
```
### Experience replay
Create pool for storing experience
```
>>> pool = MemoryPool(100000)
```
Add data to pool
```
>>> pool.add(x=[0, 1, 0, 1, 0], action=2, reward=1, next_state=[0, 0.5, 0, 0, 0.5], priority=0.9)
```
Sample data from experience pool
```
>>> pool.sample(5)
```


## Algorithms
* [DQN](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
* [DoubleDQN](https://arxiv.org/pdf/1509.06461.pdf)
* [DuelingDQN](https://arxiv.org/pdf/1511.06581.pdf)
* DuelingDDQN (Double + Dueling)
* [ActorCritic](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)
* [Priorized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)


## License

This project is licensed under the MIT License - see the LICENSE.md file for details
