import torch
import random
import numpy as np
import cv2
from collections import deque
from robinet import *
from model import Conv_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_frames = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Conv_QNet(1, [256, 128, 64], 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.n_frames
        final_action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()
            final_action[action] = 1

        return final_action


def train():
    plot_reward = []
    plot_mean_reward = []
    total_reward = 0
    record = 0
    agent = Agent()
    robinet = Robinet(180)

    # start checking if the user click for a in-reward
    mapping = {"q": -1, "w": -0.5, "e": 0, "r": 0.5, "t": 1}
    key = cv2.waitKey(10)
    if key != -1:
        char = chr(key & 0xFF)
        if char == 'd':
            key = cv2.waitKey(0)
            while not chr(key) in mapping:
                key = cv2.waitKey(0)

            robinet.set_done_reward(mapping[chr(key)])

        elif char in mapping:
            robinet.set_in_reward(mapping[char])

    while True:
        # get old state
        state_old = robinet.get_state()

        # get action
        final_action = agent.get_action(state_old)

        # perform action and get new state
        reward, done = robinet.do_step(final_action)
        state_new = robinet.get_state()

        # train short memory
        agent.train_short_memory(state_old, final_action, reward, state_new, done)

        # remember
        agent.remember(state_old, final_action, reward, state_new, done)

        if done:
            # train long memory, plot result
            robinet.reset()
            agent.n_frames += 1
            agent.train_long_memory()

            if reward > record:
                record = reward
                agent.model.save()

            print('frames : ', agent.n_frames, 'reward', reward, 'Record:', record)

            plot_reward.append(reward)
            total_reward += reward
            mean_reward = total_reward / agent.n_frames
            plot_mean_reward.append(mean_reward)
            plot(plot_reward, plot_mean_reward)


if __name__ == '__main__':
    train()
