#!/usr/bin/env python3
"""
    Play a game with desired enviroment and network.
"""
import time
import argparse
import torch
import numpy as np

import model
import wrappers

ENV_NAME = "BreakoutNoFrameskip-v4"


def state_to_torch(state):
    """
        Transform state into correct format for network.
        Args:
            state (numpy array):    Game state.
    """
    state = np.array(state)/255.0
    state = state.transpose((2, 0, 1))
    state = torch.Tensor([state]).float()
    return state


def play(env, dqn, render=False, device="cpu"):
    """
       Simulate an environment using a DQN.
        Args:
            env (gym Enviroment):   Environment from OpenAi gym.
            dqn (Pytorch Module):   Pytorch network.
            render (boolean):       Set to true to render environemnt.
                                    Default: false.
            device: (device):       Device to run the network on.
                                    Default: cpu.
        Output:
            (total_reward):         Total reward accumulated from game
                                    environment.

    """
    wait_time = 0.033
    total_reward = 0.0
    state = env.reset()
    finished = False
    while not finished:
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

    device = torch.device("cuda:0" if args.cuda else "cpu")

    env = wrappers.make_atari(args.env)
    env = wrappers.wrap_deepmind(env, False, False, True)

    net = model.DQN(4, env.action_space.n).to(device)
    net.load_state_dict(torch.load(args.model))

    score = play(env, net, True, device)
    print(f"Score: {score}")
