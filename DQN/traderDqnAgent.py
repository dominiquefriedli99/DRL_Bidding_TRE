import collections
import os
import random
from typing import Deque
from xmlrpc.client import Boolean

import numpy as np
from itertools import cycle, count

from traderDqn import DQN
from Env import Env
from tqdm import tqdm
import utils

# PROJECT_PATH = os.path.abspath("C:/Python-Projekte/TRE_Bidding/AdvancedAnalytics-DeepDive-TRERL/")
# MODELS_PATH = os.path.join(PROJECT_PATH, "models")
path = './models/'
MODEL_PATH = os.path.join(path, "tre_bidder.h5")

class Agent:
    def __init__(self, env: Env, **kwargs):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.shape
        self.actions = len(self.env.action_space)
        self.n = env.n
        self.n_successfull_bids = env.n_successfull_bids
        # DQN Agent Variables
        # tuned hyperparameters: 10.5.2022
        defaultKwargs = { "learning_rate": 1e-5, "epsilon_decay": 0.95, "tau": 0.9, "epsilon_min": 0.01, "gamma":0.95, "train_start":10_000, "replay_buffer_size": 10_000, "batch_size": 128 , "num_update_steps": 1000 }
        kwargs = defaultKwargs | kwargs

        #add tuned parameters as default params, they can be overwritten
        self.learning_rate = kwargs['learning_rate']
        self.epsilon_decay = kwargs['epsilon_decay']
        self.epsilon_min = kwargs['epsilon_min'] # 0.01 # tuned
        self.tau = kwargs['tau'] # .1 # # tuned: update rate for target_model
        self.replay_buffer_size = kwargs['replay_buffer_size'] # 10_000 # tuned
        self.train_start = kwargs['train_start']  #10_000 # tuned
        self.gamma = kwargs['gamma'] # Discountfactor, to caluclate rewards
        self.batch_size = kwargs['batch_size']
        self.num_update_steps = kwargs['num_update_steps']
        self.epsilon = 1.0
        #self.epsilon_min = 0.01 # tuned
        #self.epsilon_decay = 0.95 # tuned
        # DQN Network Variables
        self.memory: Deque = collections.deque(maxlen=self.replay_buffer_size)
        self.state_shape = self.observations
        #self.learning_rate = 1e-4 # tuned
        self.axpo_rewards = []
        self.random_rewards = []
        self.insider_rewards = []
        
        self.dqn = DQN(
            self.state_shape,
            self.actions,
            self.learning_rate,
            self.tau
        )
       
        self.target_dqn = DQN(
            self.state_shape,
            self.actions,
            self.learning_rate,
            self.tau
        )
        self.target_dqn.update_model(self.dqn)

    def get_action(self, state: np.ndarray, random: Boolean = False):
        if np.random.rand() <= self.epsilon or random:
            return np.random.randint(self.actions)
        else:
            value_function = self.dqn(state)
            return np.argmax(value_function)

    def train(self, num_episodes: int):
        self.axpo_rewards = []
        self.take_1_2_waterprice = []
        self.take_1_5_waterprice = []
        self.take_2_waterprice = []
        p_bar = tqdm(total=(self.n * num_episodes))
        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            '''take_1_5_waterprice_reward = 0.0
            take_1_2_waterprice_reward = 0.0
            take_2_waterprice_reward = 0.0'''
            state = self.env.reset() # 1. reset
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            for step in count():
                action = self.get_action(state) # get action, with epsilon greedy
                next_state, reward, done = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                
                # show other agents behaviour
                ''' reward_1_2 = self.env.getRewardForFixBehaviour(1.2)
                take_1_2_waterprice_reward += reward_1_2

                reward_1_5 = self.env.getRewardForFixBehaviour(1.5)
                take_1_5_waterprice_reward += reward_1_5

                reward_2 = self.env.getRewardForFixBehaviour(2)
                take_2_waterprice_reward += reward_2'''

                self.remember(state, action, reward, next_state, done) # schreib in buffer
                self.replay() # train, put state and values into the NN
                total_reward += reward
                state = next_state
                #p_bar.set_description('Reward: {:.2f} episode: {}, step: {}'.format(total_reward, episode, step))
                if step%self.num_update_steps == 0:
                    self.target_dqn.update_model(self.dqn)
                if done:
                    self.axpo_rewards.append(total_reward)
                    ''' self.take_1_2_waterprice.append(take_1_2_waterprice_reward)  
                    self.take_1_5_waterprice.append(take_1_5_waterprice_reward)  
                    self.take_2_waterprice.append(take_2_waterprice_reward) '''      
                    # add total_rewards to create PDF
                    # utils.save((self.axpo_rewards))
                    # save the current weights
                    self.dqn.save_model(MODEL_PATH)
                    break
                #p_bar.update(1)
        p_bar.close()
        return self.axpo_rewards
        #, self.take_1_2_waterprice, self.take_1_5_waterprice, self.take_2_waterprice

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        # if the buffer reaches minimal size, start to train, before it's only crab
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        # DDQN specific learning..
        states = np.concatenate(states).astype(np.float32) # alte policy
        states_next = np.concatenate(states_next).astype(np.float32)
        # get values by states (ts) in online network
        q_values = self.dqn(states) # get q_values from states (max or sampling, it depends the strategy)
        # get values of states(ts + 1) in target network
        q_values_next = self.target_dqn(states_next)
        
        # difference between q_values and q_values_next should be minimized
        for i in range(self.batch_size): # i: batch
            a = actions[i]
            done = dones[i]
            if done:
                # if done, just take the rewards of last
                q_values[i][a] = rewards[i]
            else:
                # Policy update with SARSA max. 
                q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i]) 
        # train the online model with updated values from target model (a littel bit like Supervised learning here)
        self.dqn.fit(states, q_values)

    '''
    def bid(self, num_episodes: int):
        self.dqn.load_model(MODEL_PATH)
        for episode in range(1, num_episodes + 1):
            axpo_total_reward = 0.0
            random_total_reward = 0.0
            insider_total_reward= 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            while True:
                action = self.get_action(state)
                random_action = self.get_action(state, True)
                next_state, reward, done = self.env.step(action)
                random_reward = self.env.getReward(random_action)
                random_total_reward += random_reward
                insider_reward = self.env.getBestReward()
                insider_total_reward += insider_reward
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                axpo_total_reward += reward
                state = next_state
                if done:
                    print(f"Cumulated revenue with TRE Bidding optimization: { axpo_total_reward:.2f}")
                    print(f"Cumulated revenue just random: {random_total_reward:.2f}")
                    print(f"Cumulated revenue possible with insider knowledge: {insider_total_reward:.2f}")
                    break'''

    def bid(self, num_episodes: int):
        self.dqn.load_model(MODEL_PATH)
        p_reward = 0.0
        for episode in range(1, num_episodes + 1):
            count_our_bid_activated = 0
            axpo_total_reward = 0.0
            take_1_5_waterprice_reward = 0.0
            take_1_2_waterprice_reward = 0.0
            take_2_waterprice_reward = 0.0
            insider_total_reward= 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            while True:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                axpo_total_reward += reward
                if reward > 0:
                    count_our_bid_activated += 1
                
                # random
                '''random_action = self.get_action(state, True)
                random_reward = self.env.getReward(random_action)
                random_total_reward += random_reward'''
                
                # best possible
                insider_reward = self.env.getBestReward()
                insider_total_reward += insider_reward

                reward_1_2 = self.env.getRewardForFixBehaviour(1.2)
                take_1_2_waterprice_reward += reward_1_2

                reward_1_5 = self.env.getRewardForFixBehaviour(1.5)
                take_1_5_waterprice_reward += reward_1_5

                reward_2 = self.env.getRewardForFixBehaviour(2)
                take_2_waterprice_reward += reward_2

                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                state = next_state
                if done:
                    #print(f"Cumulated revenue with TRE Bidding optimization: { axpo_total_reward:.2f}")
                    #print(f"Cumulated revenue just random: {random_total_reward:.2f}")
                    #print(f"Cumulated revenue possible with insider knowledge: {insider_total_reward:.2f}")
                    break
        
        if count_our_bid_activated > 0: 
            p_reward = count_our_bid_activated / self.n_successfull_bids
        
        return (round(axpo_total_reward, 2), round(insider_total_reward, 2), round(take_1_2_waterprice_reward, 2), round(take_1_5_waterprice_reward, 2), round(take_2_waterprice_reward, 2), self.n_successfull_bids, round(p_reward, 2))
