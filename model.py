import numpy as np

import torch
import torch.nn as nn


class DQN(nn.Module):
    """ The model architecture as presented in doi:10.1038/nature14236
            Args:
                input_channels (int):       number of channels in the input
                n_actions (int):            number of actions the network
                                            can perform
                input_image_size (tuple):   a tuple consisting of the width and
                                            height of the imput image.
                                            Default=(84, 84)
    """
    def __init__(self, input_channels, n_actions, input_image_size=(84, 84)):
        super().__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def calc_fc_input_size(channels, image_size):
            sample = torch.zeros((1, channels) + image_size)
            sample = self.conv1(sample)
            sample = self.conv2(sample)
            sample = self.conv3(sample)
            return np.prod(sample[0].shape)

        fc_input_len = calc_fc_input_size(input_channels, input_image_size)
        self.fc1 = nn.Linear(fc_input_len, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
