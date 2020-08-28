"""
    This file contains the class experience buffer, and its integral methods,
    used for replay memory.
"""

import random
import numpy as np

import torch

from utils import state_to_torch


class CircularList():
    """ Circular list: replaces oldest elements if filled"""
    def __init__(self, max_size):
        self._list = [None for _ in range(max_size)]
        self._max_size = max_size
        self._size = 0
        self.index = 0

    def append(self, item):
        """ Append object to the item at index self.index. """
        self._list[self.index] = item
        self._size = min(self._size+1, self._max_size)
        self.index = (self.index + 1) % self._max_size

    def __getitem__(self, idx):
        return self._list[idx % self._max_size]

    def __len__(self):
        return self._size


class ExperienceBuffer():
    """ Experience buffer used for replay memory."""
    def __init__(self, max_size):
        self.buffer = CircularList(max_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        """ Append object to buffer. """
        self.buffer.append(experience)

    def get_random_sample(self, sample_size):
        """ get a random sample from buffer.
                Args:
                    sample_size (int): size of the sample"""
        # random.sample is uglier but also much faster than random.choice
        indices = random.sample(range(len(self.buffer)), sample_size)
        states, actions, rewards, dones, next_states =\
            zip(*[self.buffer[i] for i in indices])
        dones = np.array(dones, dtype=np.uint8)
        return state_to_torch(states), \
            torch.tensor(actions), torch.Tensor(rewards), \
            torch.tensor(dones, dtype=torch.bool), \
            state_to_torch(next_states)
