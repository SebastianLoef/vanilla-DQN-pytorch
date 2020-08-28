"""
    All general parameters and hyperparameters for training.
    Hyperparameters are from: https://www.nature.com/articles/nature14236
"""

# General parameters
ENV_NAME = "BreakoutNoFrameskip-v4"
PLAY_FRAME_LIMIT = 10000  # max number of frames in test_network.play
GRAPHICS_CARD = "cuda:0"

# Hyperparameters
MAX_FRAMES = 50 * 10**6
# (Which frames to save network during training)
SAVE_NETWORK_FRAMES = [0, 10**5, 5*10**5, 10**6, 5*10**6, 10**7, 25*10**6]
NETWORK_SAVE_LOCATION = "./networks/"

MINIBATCH_SIZE = 32

GAMMA = 0.999  # Discount factor
UPDATE_FREQ = 4
LEARNING_RATE = 0.000025
GRAD_MOMENTUM = 0.95
SQ_GRAD_MOMENTUM = 0.95
MIN_SQ_GRAD = 0.01

EPSILON_INITIAL = 1.0
EPSILON_FINAL = 0.05
EPSILON_FINAL_FRAME = 10**6

REPLAY_START_SIZE = 50000
REPLAY_MEMORY_SIZE = 10**6
AGENT_HIST_LENGTH = 4
TGT_NET_UPDATE_FREQ = 10000
