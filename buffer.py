import random
import numpy as np

import torch


def state_to_torch(state):
    state = np.array(state)/255.0
    if len(state.shape) == 3:
        state = [state.transpose((2, 0, 1))]
    else:
        state = state.transpose((0, 3, 1, 2))
    state = torch.Tensor(state).float()
    return state


class CircularList():
    def __init__(self, max_size):
        self._list = [None for _ in range(max_size)]
        self._max_size = max_size
        self._size = 0
        self._index = 0

    def append(self, object):
        self._list[self._index] = object
        self._size = min(self._size+1, self._max_size)
        self._index = (self._index + 1) % self._max_size

    def __getitem__(self, idx):
        return self._list[idx % self._max_size]

    def __len__(self):
        return self._size


class ExperienceBuffer():
    def __init__(self, max_size):
        self.buffer = CircularList(max_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def get_random_batch(self, batch_size):
        # random.sample is uglier but also much faster than random.choice
        indices = random.sample(range(len(self.buffer)), batch_size)
        states, actions, rewards, dones, next_states =\
            zip(*[self.buffer[i] for i in indices])
        dones = np.array(dones, dtype=np.uint8)
        return state_to_torch(states), \
            torch.tensor(actions), torch.Tensor(rewards), \
            torch.tensor(dones, dtype=torch.bool), \
            state_to_torch(next_states)
