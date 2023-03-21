from collections import deque
import copy
from multiprocessing import Process, freeze_support
import os
import sys
import time

import numpy as np
from homework2 import Hw2Env
import torch.nn as nn
import torch

from utils.hw2.replay_buffer import ReplayBuffer

INITIAL_TEMPERATURE = 500

def startTraining(model: nn.Module, batch_size=128, gamma=0.99, device = torch.device):
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    if os.path.exists('./models/hw2.pt'):
        model.load_state_dict(torch.load('./models/hw2.pt'))
    target_model = copy.deepcopy(model).to(device)
    replay_buffer = ReplayBuffer(1000)

    episode = 0
    step = 0

    rewards = deque(maxlen=100)

    temperature = INITIAL_TEMPERATURE

    def updateModel(done: bool):
        if len(replay_buffer) > batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
            q_values = model(torch.tensor(state_batch, dtype=torch.float32).to(device)).gather(1, torch.tensor(action_batch, dtype=torch.int64).to(device).unsqueeze(1))
            target_q_values = torch.tensor(reward_batch,  dtype=torch.float32).to(device)  + gamma * torch.max(target_model(torch.tensor(next_state_batch, dtype=torch.float32).to(device)), dim=1)[0] * (torch.tensor(~done_batch, dtype=torch.bool).to(device))
            loss = loss_fn(q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if(step % 10 == 0):
            target_model.load_state_dict(model.state_dict())
            sys.stdout.write(f"Step: {step:4d}     Reward: {np.mean(rewards):.5f}     Temperature: {temperature:.1f}\r")
            sys.stdout.flush()
        if done and episode % 5 == 0:
            torch.save(model.state_dict(), './models/hw2.pt')
            print(f"Episode: {episode}\tReward: {np.mean(rewards):.5f}                                        ")
   
    
    env = Hw2Env(n_actions=8, render_mode="gui")
    env.reset()
    state = env.state()
    while True:
        action_probs = torch.softmax(model(state.unsqueeze(0).to(device))/temperature, dim=1).squeeze(0).detach().cpu().numpy()
        action = np.random.choice(len(action_probs), p=action_probs)
        state, action, reward, next_state,  done, = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        rewards.append(reward)
        if done:
            episode += 1
            env.reset()
            state = env.state()
        step += 1
        if(temperature > 0.1):
            temperature -= 0.1
        for i in range(5):
            updateModel(False)
        if(done):
            updateModel(True)