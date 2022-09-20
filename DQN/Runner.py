
import pandas as pd
import numpy as np
from traderDqnAgent import Agent
import utils;
import matplotlib.pyplot as plt
from Env import Env
import collections
from typing import Deque

class Runner:

     def store_params(self, **kwargs):
          # write the currently best parameters who gives the best accuracy into the file
          # overwrite the file
          with open("C:\\temp\\best_params_DQN.txt", 'w') as f:
               for kw in kwargs:
                    f.write('{}: {:.5f}\n'.format(kw, kwargs[kw]))


     def plot(self, df, learning_rates, color_map, num_episodes_train, filename):
          plot_result = []
          colors = [color_map(i) for i in np.linspace(0, 1, len(learning_rates))]
          dqn_agent = Agent(Env(df))
          for i, lr in enumerate(learning_rates):
               dqn_agent.learning_rate = lr
               data = dqn_agent.train(num_episodes_train)
               res_tuple = (lr, colors[i], data)
               plot_result.append(res_tuple)                       
          utils.save(plot_result, filename)
     
     def plot_dynamic(self, df, arg_name, arguments, color_map, num_episodes_train, filename, label):
          plot_result = []
          env = Env(df)
          colors = [color_map(i) for i in np.linspace(0, 1, len(arguments))]
          for i, value in enumerate(arguments):
               keywords = { arg_name: value }
               dqn_agent = Agent(env, **keywords)
               data = dqn_agent.train(num_episodes_train)
               res_tuple = (value, colors[i], data)
               plot_result.append(res_tuple)                       
          utils.save(plot_result, filename, label)

     def plot_dynamic_agents(self, df, num_episodes_train, color_map, labels, filename):
          plot_result = []
          env = Env(df)
          colors = [color_map(i) for i in np.linspace(0, 1, len(labels))]
          dqn_agent = Agent(env)
          data = dqn_agent.train(num_episodes_train)
          for i, label in enumerate(labels):
               res_tuple = (colors[i], data[i], label)
               plot_result.append(res_tuple)                       
          utils.save(plot_result, filename)

     def plot_results(self, train_data, test_data, num_episodes_train):
          '''test_data.sort_values(by='datetime_start_utc', ascending=True)
          start_date = test_data.iloc[0]['datetime_start_utc']
          end_date = test_data.iloc[len(test_data) -1]['datetime_start_utc']
          filename = 'DQN_Bidding_results_from_{}_until_{}.csv'.format(start_date, end_date)'''
          result_df = pd.DataFrame(columns=['replay_buffer_size','train_start','gamma', 'tau','epsilon_min','epsilon_decay','learning_rate','num_episodes_train', 'accuracy', 
          'cumulated_reward', 'best_possible_reward', 'take_1_2_waterprice_reward', 'take_1_5_waterprice_reward', 'take_2_waterprice_reward', 'number_activated_hours'])  
          # tune Hyperparameters TEST
          replay_buffer_sizes = [10_000]
          train_starts = [10_000]
          gammas = [0.95]
          taus = [0.9]
          epsilon_mins = [0.01]
          epsilon_decays = [0.95]
          learning_rates = [1e-4, 1e-5]
          '''
          replay_buffer_sizes = [10_000]
          train_starts = [10_000]
          gammas = [0.99]
          taus = [1.0]
          epsilon_mins = [0.001]
          epsilon_decays = [0.95]
          learning_rates = [1e-4]
          '''
          best_cum_reward = 0
          for replay_buffer_size in replay_buffer_sizes:
               for train_start in train_starts:
                    for gamma in gammas:
                         for tau in taus:
                              for epsilon_min in epsilon_mins:
                                   for epsilon_decay in epsilon_decays:
                                        for lr in learning_rates:
                                             train_agent = Agent(Env(train_data))
                                             # add hyperparameters to train_agent
                                             train_agent.replay_buffer_size = replay_buffer_size
                                             train_agent.train_start = train_start
                                             train_agent.memory: Deque = collections.deque(maxlen=train_agent.replay_buffer_size)
                                             train_agent.gamma = gamma # Discountfactor, to caluclate rewards
                                             train_agent.tau = tau # update rate for target_model
                                             train_agent.epsilon_min = epsilon_min
                                             train_agent.epsilon_decay = epsilon_decay
                                             train_agent.learning_rate = lr
                                             train_agent.train(num_episodes_train)
                                             # create axpo_agent
                                             axpo_agent = Agent(Env(test_data))
                                             
                                             axpo_agent.epsilon = 0
                                             result = axpo_agent.bid(1)
                                             
                                             entry = pd.DataFrame({
                                                  'replay_buffer_size': [replay_buffer_size],
                                                  'train_start': [train_start], 
                                                  'gamma': [gamma], 
                                                  'tau': [tau], 
                                                  'epsilon_min': [epsilon_min],
                                                  'epsilon_decay': [epsilon_decay], 
                                                  'num_episodes_train': [num_episodes_train], 
                                                  'learning_rate': [lr], 
                                                  'cumulated_reward': [result[0]], 
                                                  'accuracy': [result[6]],
                                                  'best_possible_reward':[result[1]],
                                                  'take_1_2_waterprice_reward':[result[2]],
                                                  'take_1_5_waterprice_reward':[result[3]],
                                                  'take_2_waterprice_reward':[result[4]],
                                                  'number_activated_hours': [result[5]]})
                                             
                                             result_df = pd.concat([result_df, entry])
                                             cum_reward = result[0]
                                             if cum_reward > best_cum_reward:
                                                  best_cum_reward = cum_reward
                                                  print("bether cumulated reward found. ", best_cum_reward)
                                                  self.store_params(cumulated_reward = best_cum_reward, rbf = replay_buffer_size, ts = train_start, tau = tau, eps_min = epsilon_min, eps_decay=epsilon_decay, lr = lr)
                                        
          result_df.to_csv("C:\\temp\\result_tuned_train_and_bid_20episodes_DQN.csv", sep=";")
