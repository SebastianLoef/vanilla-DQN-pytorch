""" Common utilites """
import numpy as np

import torch


def state_to_torch(state):
    """
        Convert state to torch tensor.
    """
    state = np.array(state)/255.0
    if len(state.shape) == 3:
        state = [state.transpose((2, 0, 1))]
    else:
        state = state.transpose((0, 3, 1, 2))
    state = torch.Tensor(state).float()
    return state
