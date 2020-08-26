#!/usr/bin/env python3
"""
    Training network. All parameters can be found directly under imported
    modules. All the currently set parameters are the same as described in
    DeepMinds paper Human-level control through deep reinforcement learning.
    https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
"""
import argparse
import time
import random
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import wrappers
import model
from buffer import ExperienceBuffer

ENV_NAME = "BreakoutNoFrameskip-v4"
MAX_FRAMES = 10*10**6
SAVE_NETWORK_FRAMES = [0, 10**5, 5*10**5, 10**6, 5*10**6, 10**7, 25*10**6]

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

Experience = collections.namedtuple('Experience', ['state', 'action', 'reward',
                                                   'is_done', 'new_state'])


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


class Agent():
    """
        Agent playing step by step in environment and storing experiences in
        experience buffer.
        Args:
            env (Environment):  Atari environment
            exp_buffer (ExperienceBuffer):  experience buffer defined in
                                            ./buffer.py
    """
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon, device="cpu"):
        done_reward = None
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_v = state_to_torch(self.state)
                state_v = state_v.to(device)
                q_values = net(state_v)
                action_v = q_values.max(1)[1]
                action = int(action_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done,
                         new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


criterion = nn.MSELoss()


def calc_action_values(net, tgt, batch, device="cpu"):
    """ Calculate state action values from given batch. """
    states, actions, rewards, dones, next_states = batch
    states_v = states.to(device)
    actions_v = actions.to(device)
    rewards_v = rewards.to(device)
    done_mask = dones.to(device)
    next_states_v = next_states.to(device)

    predicted_state_action_values =\
        net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    expected_state_action_values = expected_state_action_values.float()
    predicted_state_action_values = predicted_state_action_values.float()
    return predicted_state_action_values, expected_state_action_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Enable cuda")
    parser.add_argument("--env", default=ENV_NAME,
                        help="Name of the environment, default=" + ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda:1" if args.cuda else "cpu")

    env = wrappers.make_atari(args.env)
    env = wrappers.wrap_deepmind(env, episode_life=False, frame_stack=True)
    exp_buffer = ExperienceBuffer(REPLAY_MEMORY_SIZE)
    agent = Agent(env, exp_buffer)

    net = model.DQN(AGENT_HIST_LENGTH, env.action_space.n).to(device)
    tgt_net = model.DQN(AGENT_HIST_LENGTH, env.action_space.n).to(device)
    tgt_net.load_state_dict(net.state_dict())

    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE,
                              momentum=GRAD_MOMENTUM, eps=MIN_SQ_GRAD)

    writer = SummaryWriter(comment="-" + args.env)

    remaining_time_buffer = collections.deque(maxlen=100)
    last_100_rewards = collections.deque(maxlen=100)

    episode_idx = 0
    frame_idx = 0
    while frame_idx < MAX_FRAMES:
        episode_t = time.time()
        frame_idx_old = frame_idx
        total_loss = 0.0
        done_reward = None
        while done_reward is None:
            eps = max(EPSILON_FINAL,
                      EPSILON_INITIAL + (EPSILON_FINAL-EPSILON_INITIAL)
                      / EPSILON_FINAL_FRAME*frame_idx)
            done_reward = agent.play_step(net, eps, device)

            if frame_idx in SAVE_NETWORK_FRAMES:
                torch.save(net.state_dict(), args.env + "-frame-" +
                           str(frame_idx) + ".dat")

            frame_idx += 1
            if len(exp_buffer) < REPLAY_START_SIZE:
                continue

            if frame_idx % TGT_NET_UPDATE_FREQ == 0:
                tgt_net.load_state_dict(net.state_dict())

            if frame_idx % UPDATE_FREQ != 0:
                continue

            optimizer.zero_grad()
            batch = exp_buffer.get_random_batch(MINIBATCH_SIZE)
            action_values = calc_action_values(net, tgt_net, batch, device)
            loss = criterion(*action_values)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        d_frames = frame_idx-frame_idx_old

        average_loss = total_loss/d_frames

        fps = int(d_frames/(time.time()-episode_t))
        remaining_time_buffer.append((MAX_FRAMES-frame_idx)/(fps*3600))
        remaining_time = np.mean(remaining_time_buffer)

        last_100_rewards.append(done_reward)
        mean_reward = np.mean(last_100_rewards)

        print(f"Episode: {episode_idx}, "
              f"total frames: {frame_idx}, "
              f"mean reward 100 games: {mean_reward:.2f}, "
              f"speed: {fps} f/s, "
              f"remaining time: {remaining_time:.2f} h")

        writer.add_scalar("average_loss", average_loss, frame_idx)
        writer.add_scalar("reward", done_reward, frame_idx)
        writer.add_scalar("mean_reward_100", mean_reward, frame_idx)
        writer.add_scalar("epsilon", eps, frame_idx)
        writer.add_scalar("frames_per_episode", d_frames, episode_idx)
        writer.add_scalar("speed", fps, frame_idx)
        episode_idx += 1

    writer.close()
    torch.save(net.state_dict(), args.env + "-final.dat")
