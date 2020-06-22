#!/usr/bin/env python3
import wrappers
import model

import argparse
import gym
import time
import random
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "BreakoutNoFrameskip-v4"
MAX_FRAMES = 2*10**6
SAVE_NETWORK_FRAMES = [0, 10**5, 2*10**5, 10**6, 5*10**6, 10**7, 25*10**6]

MINIBATCH_SIZE = 32

GAMMA = 0.99 # Discount factor
ACTION_REPEAT = 4
UPDATE_FREQ = 4

LEARNING_RATE = 0.00025
GRAD_MOMENTUM = 0.95
SQ_GRAD_MOMENTUM = 0.95
MIN_SQ_GRAD = 0.01

EPSILON_INITIAL = 1.0
EPSILON_FINAL = 0.1
EPSILON_FINAL_FRAME = 10**6

REPLAY_START_SIZE = 50000
REPLAY_MEMORY_SIZE = 10**6
AGENT_HIST_LENGTH = 4
TGT_NET_UPDATE_FREQ = 10000

Experience = collections.namedtuple('Experience', ['state', 'action', 'reward',
                                                   'is_done', 'new_state'])

class clist():
    def __init__(self, max_size):
        self._list = [None]*max_size
        self._max_size = max_size
        self._size = 0
        self._index = 0
    
    def append(self, object):
        return_idx = self._index
        self._list[self._index] = object
        self._size = min(self._size+1, self._max_size)
        self._index = (self._index + 1) % self._max_size
        return return_idx

    def __getitem__(self, idx):
        if self._size > 0:
            return self._list[idx % self._size]
        else:
            return None

    def __len__(self):
        return self._size

    def __setitem__(self, idx, value):
        assert idx < self._max_size, "index out of range"
        self._list[idx] = value

    def index(self, value, start, stop):
        index = None
        for i in range(start, stop):
            if np.all(value == self.__getitem__(i)):
                index = i
        return index


class LazyFrames():
    def __init__(self, max_size):
        self.buffer = clist(max_size)
        self.last_frame = 0
        self.hist_check_len = AGENT_HIST_LENGTH - 1

    def append(self, frame):
        """
            Only adds frame if it does not already exist among previous frames.
        """
        start = self.last_frame-self.hist_check_len
        stop = self.last_frame
        obj_idx = self.buffer.index(frame, start, stop)

        if not obj_idx:
            obj_idx = self.buffer.append(frame)
        else:
            obj_idx = self.last_frame
        return obj_idx

    def __getitem__(self, idx):
        """ Return stack of frames """
        frames = [self.buffer[i] 
                 for i in range(idx, idx+AGENT_HIST_LENGTH)]
        return np.stack(frames)

class LazyFrameStack():
    def __init__(self, max_size):
        self.state_indexes = clist(max_size+1)
        self.states = LazyFrames(max_size+AGENT_HIST_LENGTH)

    def append(self, state): 
        index = None
        for frame in state: 
            index = self.states.append(frame) if index is None else index

        return_idx = self.state_indexes.append(index)
        return return_idx 

    def __getitem__(self, idx):
        state_idx = self.state_indexes[idx]
        state = self.states[state_idx]
        return state
        
class ExperienceBuffer():
    def __init__(self, max_size):
        self.buffer = clist(max_size)
        self.state_buffer = LazyFrameStack(max_size)
 
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        state, action, reward, is_done, new_state = experience
        state_idx = self.state_buffer.append(state)
        new_state_idx = self.state_buffer.append(new_state)
        lazy_experience = Experience(self.state_buffer[state_idx],
                                     action, reward, is_done,
                                     self.state_buffer[new_state_idx])
        self.buffer.append(lazy_experience)

    def get_random_batch(self, batch_size):
        indices = random.sample(range(len(self.buffer)), batch_size)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)


    
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
                state_v = torch.Tensor([self.state]).to(device)
                q_values = net(state_v)
                action_v = q_values.max(1)[1]

                action = int(action_v.item())
        
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        
        exp = Experience(self.state.astype(np.uint8), action, reward, is_done,
                         new_state.astype(np.uint8))
        self.exp_buffer.append(exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

criterion = nn.MSELoss()

def calc_action_values(net, tgt, batch, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states).float().to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)
    next_states_v = torch.tensor(next_states).float().to(device)

    predicted_state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return predicted_state_action_values, expected_state_action_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Enable cuda")
    parser.add_argument("--env", default=ENV_NAME,
                        help="Name of the environment, default=" + ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda:1" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)
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
            eps = max(EPSILON_FINAL, EPSILON_INITIAL + (EPSILON_FINAL-EPSILON_INITIAL)/EPSILON_FINAL_FRAME*frame_idx)
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

        d_frames =  frame_idx-frame_idx_old
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
