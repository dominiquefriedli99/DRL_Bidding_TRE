import collections
import os
import random
from typing import Deque
from xmlrpc.client import Boolean

import numpy as np
from itertools import cycle, count
from Env import Env
import utils
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from traderBranchingDqn import BranchingQNetwork, ExperienceReplayMemory
path = './models/'
class Agent:
    def __init__(self, env: Env, **kwargs):
        # DQN Env Variables
        self.env = env
        self.observations = len(env.observation_space)
        self.actions = len(self.env.action_space)
        self.state_shape = env.observation_space.shape
        self.bins = env.bins
        self.n = env.n
        self.n_successfull_bids = env.n_successfull_bids
        # DQN Agent Variables
        # tuned hyperparameters: 10.5.2022
        defaultKwargs = { "learning_rate": 0.0001, "epsilon_decay": 0.99, "epsilon_min": 0.1, "gamma":0.9, "learning_start": 20_000, "num_update_steps": 10_000, "batch_size": 256 ,  "replay_buffer_size": 50_000 }
        kwargs = defaultKwargs | kwargs

        #add tuned parameters as default params, they can be overwritten
        self.learning_rate = kwargs['learning_rate']
        self.epsilon_decay = kwargs['epsilon_decay']
        self.epsilon_min = kwargs['epsilon_min'] # 0.01 # tuned
        self.replay_buffer_size = kwargs['replay_buffer_size'] # 10_000 # tuned
        self.learning_start = kwargs['learning_start']  #10_000 # tuned
        self.gamma = kwargs['gamma'] # Discountfactor, to caluclate rewards
        self.batch_size = kwargs['batch_size']
        self.num_update_steps = kwargs['num_update_steps']
        self.epsilon = 1.0

        self.memory = ExperienceReplayMemory(self.replay_buffer_size)
        self.q = BranchingQNetwork(self.observations, self.actions, self.bins)
        self.target = BranchingQNetwork(self.observations, self.actions, self.bins)
        self.target.load_state_dict(self.q.state_dict())
        self.update_counter = 0
        self.adam = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        self.axpo_rewards = []
        self.random_rewards = []
        self.insider_rewards = []
    
    # should return numpy array with actions of len(actions) = 13
    def get_action(self, state):
        with torch.no_grad():
            # a = self.q(x).max(1)[1]
            out = self.q(state).squeeze(0)
            action = torch.argmax(out, dim=1)
        return action.numpy()

    def select_action(self, state):
        if np.random.random() > self.epsilon:
            return self.get_action(state)
        else:
            # size is a int
            return np.random.randint(0, self.bins, size=self.actions)

    def train(self, num_episodes: int):
        # self.cumulated_effectiv_return = []
        self.total_effectiv_return_ts = []
        # self.cumulated_reward = []
        i = 0
        p_bar = tqdm(total=(self.n * num_episodes))
        for episode in range(1, num_episodes + 1):
            #total_reward = 0.0 # reward
            total_axpo_effectiv_return = 0.0 # effective return in €
            state = self.env.reset() # 1. reset
            for step in count():
                i += 1
                action = self.select_action(state) # get action, with epsilon greedy
                next_state, reward, done, _, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done) # write into buffer, decay epsilon
                #total_reward += reward
                total_axpo_effectiv_return += reward
                state = next_state
                p_bar.set_description('Effectiv return is: {:.2f} episode: {}, step: {}'.format(total_axpo_effectiv_return, episode, step))
                if i > self.learning_start:
                    self.update_policy(self.adam, self.memory)
                
                if i%self.num_update_steps == 0:
                    torch.save(self.q.state_dict(), os.path.join(path, 'model_state_dict'))
                if done:
                    #self.cumulated_reward.append(total_reward)
                    #mean_profit = total_axpo_effectiv_return/self.n
                    self.total_effectiv_return_ts.append(total_axpo_effectiv_return)
                    utils.plot_progress(self.total_effectiv_return_ts, None)
                    total_axpo_effectiv_return = 0
                    torch.save(self.q.state_dict(), os.path.join(path, 'model_state_dict'))
                    break
                p_bar.update(1)
        p_bar.close()
        return self.total_effectiv_return_ts
    
    def remember(self, state, action, reward, next_state, done):
        # state.reshape(-1): Tensor
        # list
        self.memory.push((state.reshape(-1).numpy().tolist(), action, reward, next_state.reshape(-1).numpy().tolist(), 0. if done else 1.))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def bid(self, total_effectiv_return_ts):
        self.q = BranchingQNetwork(self.observations, self.actions, self.bins)
        model = os.path.join(path, 'model_state_dict')
        self.q.load_state_dict(torch.load(model))
        self.successfull_actions = []
        self.all_actions = []
        axpo_return_1_2_waterprice = 0
        axpo_return_1_7_waterprice = 0
        axpo_return_1_5_waterprice = 0
        axpo_return_2_waterprice = 0
        self.successfull_actions_1_2 = []
        self.successfull_actions_1_7 = []
        self.successfull_actions_1_5 = []
        self.successfull_actions_2 = []
        total_water_price = 0
        axpo_total_return = 0.0
        count_activated_hours = 0
        waterprice = 0
        for ep in tqdm(range(1)):     
            done = False
            state = self.env.reset()
            while not done:
                with torch.no_grad(): # kein training gerade, nur abruf vom ergebnis
                    out = self.q(state).squeeze(0)
                action = torch.argmax(out, dim=1).numpy().reshape(-1)
                state, reward, done, successfull_actions, all_actions = self.env.step(action)
               
                # prepare values to return
                axpo_total_return += reward
                if len(successfull_actions) > 0:
                    self.successfull_actions.extend(successfull_actions)
                self.all_actions.extend(all_actions) 
                if reward > 0:
                    count_activated_hours += 1
                
                # calculate benchmark
                
                waterprice = self.env.getWaterPrice()
                total_water_price  += waterprice

                result_1_2 = self.env.getRewardForFixBehaviour(1.2)
                axpo_return_1_2_waterprice += result_1_2[0]
                if len(result_1_2[1]) > 0:
                    self.successfull_actions_1_2.extend(result_1_2[1])

                result_1_5 = self.env.getRewardForFixBehaviour(1.5)
                axpo_return_1_5_waterprice += result_1_5[0]
                if len(result_1_5[1]) > 0:
                    self.successfull_actions_1_5.extend(result_1_5[1])

                result_1_7 = self.env.getRewardForFixBehaviour(1.7)
                axpo_return_1_7_waterprice += result_1_7[0]
                if len(result_1_7[1]) > 0:
                    self.successfull_actions_1_7.extend(result_1_7[1])

                result_2 = self.env.getRewardForFixBehaviour(2)
                axpo_return_2_waterprice += result_2[0]
                if len(result_2[1]) > 0:
                    self.successfull_actions_2.extend(result_2[1])

                if done:
                    utils.plot_progress(total_effectiv_return_ts, axpo_total_return)
                    break
        
        return axpo_total_return, self.successfull_actions, self.all_actions, count_activated_hours, total_water_price, axpo_return_1_2_waterprice, axpo_return_1_5_waterprice, axpo_return_1_7_waterprice, axpo_return_2_waterprice, self.successfull_actions_1_2, self.successfull_actions_1_5,  self.successfull_actions_1_7, self.successfull_actions_2
    
    def update_policy(self, adam, memory):
        # get mini-batch
        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(self.batch_size)

        states = torch.tensor(b_states).float()
        actions = torch.tensor(b_actions).long().reshape(states.shape[0], -1, 1)
        rewards = torch.tensor(b_rewards).float().reshape(-1, 1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1, 1)

        current_q_values = self.q(states).gather(2, actions).squeeze(-1)

        with torch.no_grad():

            argmax = torch.argmax(self.q(next_states), dim=2)

            max_next_q_vals = self.target(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            max_next_q_vals = max_next_q_vals.mean(1, keepdim=True).expand(-1, max_next_q_vals.shape[1])

        expected_q_vals = rewards + max_next_q_vals * self.gamma * masks # Belmann
        # print(expected_q_vals[:5])
        loss = F.mse_loss(expected_q_vals, current_q_values)

        # input(loss)

        # print('\n'*5)

        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters():
            p.grad.data.clamp_(-1., 1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.num_update_steps == 0:
            self.update_counter = 0
            self.target.load_state_dict(self.q.state_dict())
