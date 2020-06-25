#!/usr/bin/env python3
import model
import wrappers

import numpy as np
import torch
import time
import argparse

ENV_NAME = "BreakoutNoFrameskip-v4"

def state_to_torch(state):
    state = np.array(state)/255.0
    state = state.transpose((2, 0, 1))
    state = torch.Tensor([state]).float()
    return state

def play(env, model, render=False, device="cpu"):
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
        q_values = model(state_v)
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
                        help="Enable cuda")
    parser.add_argument("--env", default=ENV_NAME,
                        help="Name of the environment, default=" + ENV_NAME)
    parser.add_argument("-m", "--model", help="Model to play the game")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.cuda else "cpu")

    env = wrappers.make_atari(args.env)
    env = wrappers.wrap_deepmind(env, False, False, True)

    net = model.DQN(4, env.action_space.n).to(device)
    net.load_state_dict(torch.load(args.model))
    score = play(env, net, True, device)
    print(f"Score: {score}")

