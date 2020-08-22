#!/usr/bin/env python3
import wrappers
import model
import argparse
import time
import random
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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

    # def __setitem__(self, idx, value):
    #    assert idx < self._max_size, "index out of range"
    #    self._list[idx] = value


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
        return state_to_torch(states), \
            np.array(actions), np.array(rewards), \
            np.array(dones, dtype=np.uint8), \
            state_to_torch(next_states)


def state_to_torch(state):
    state = np.array(state)/255.0
    if len(state.shape) == 3:
        state = [state.transpose((2, 0, 1))]
    else:
        state = state.transpose((0, 3, 1, 2))
    state = torch.Tensor(state).float()
    return state


class Agent():
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
    states, actions, rewards, dones, next_states = batch
    states_v = states.to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)
    next_states_v = next_states.to(device)

    predicted_state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
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

    episode_idx = 0
    frame_idx = 0
    t_left_approx_buffer = collections.deque(maxlen=100)
    all_rewards = []
    while True:
        episode_t = time.time()
        frame_idx_old = frame_idx
        total_loss = 0.0
        done_reward = None
        while done_reward is None:
            eps = max(EPSILON_FINAL, 
                      EPSILON_INITIAL + (EPSILON_FINAL-EPSILON_INITIAL)/EPSILON_FINAL_FRAME*frame_idx)
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
            state_action_values = calc_action_values(net, tgt_net, batch, device)
            loss = criterion(*state_action_values)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        d_frames = frame_idx-frame_idx_old
        average_loss = total_loss/d_frames
        fps = int(d_frames/(time.time()-episode_t))
        t_left_approx_buffer.append((MAX_FRAMES-frame_idx)/(fps*3600))
        estimated_time_left = np.mean(t_left_approx_buffer)

        all_rewards.append(done_reward)
        mean_reward = np.mean(all_rewards[-100:])

        print(f"Episode: {episode_idx}, simulated frames: {frame_idx}, mean reward 100 games: {mean_reward:.2f},\
 speed: {fps} f/s, Estimated time left: {estimated_time_left:.2f} h")

        writer.add_scalar("average_loss", average_loss, frame_idx)
        writer.add_scalar("reward", done_reward, frame_idx)
        writer.add_scalar("mean_reward_100", mean_reward, frame_idx)
        writer.add_scalar("epsilon", eps, frame_idx)
        writer.add_scalar("frames_per_episode", d_frames, episode_idx)
        writer.add_scalar("speed", fps, frame_idx)
        episode_idx += 1
        if frame_idx > MAX_FRAMES:
            break
    writer.close()
    torch.save(net.state_dict(), args.env + "-final.dat")
