#!/usr/bin/env python3
"""
    Play a game with desired enviroment and network.
"""
import time
import argparse
import collections
from itertools import count
import numpy as np
import torch

import model
import wrappers
from utils import state_to_torch
from parameters import ENV_NAME, GRAPHICS_CARD, PLAY_FRAME_LIMIT


def play(env, dqn, render=False, device="cpu", frame_limit=PLAY_FRAME_LIMIT):
    """
       Simulate an environment using a DQN.
        Args:
            env (gym Enviroment):   Environment from OpenAi gym.
            dqn (Pytorch Module):   Pytorch network.
            render (boolean):       Set to true to render environemnt.
                                    Default: false.
            device: (device):       Device to run the network on.
                                    Default: cpu.
            frame_limit (int):      Frame limit during playthrough.
        Output:
            (total_reward):         Total reward accumulated from game
                                    environment.

    """
    wait_time = 0.033
    total_reward = 0.0
    state = env.reset()
    old_states = collections.deque(maxlen=20)
    finished = False
    for frame in count():
        old_states.append(state)
        is_stuck = np.all(np.array(old_states[0]) == np.array(state))
        if finished or is_stuck or frame < frame_limit:
            if len(old_states) >= 20:
                break

        if render:
            env.render()
            start_time = time.time()

        state_v = state_to_torch(state)
        state_v = state_v.to(device)
        q_values = dqn(state_v)
        action = q_values.max(1)[1]
        action = int(action.item())
        state, reward, finished, _ = env.step(action)
        total_reward += reward

        if render:
            time_to_sleep = wait_time - (time.time()-start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Render on graphics card(cuda:0).")
    parser.add_argument("--env", default=ENV_NAME,
                        help="Name of the environment, default=" + ENV_NAME)
    parser.add_argument("-m", "--model", help="DQN")
    args = parser.parse_args()

    device = torch.device(GRAPHICS_CARD if args.cuda else "cpu")

    env = wrappers.make_atari(args.env)
    env = wrappers.wrap_deepmind(env, False, False, True)

    net = model.DQN(4, env.action_space.n).to(device)
    net.load_state_dict(torch.load(args.model))

    score = play(env, net, True, device)
    print(f"Score: {score}")
