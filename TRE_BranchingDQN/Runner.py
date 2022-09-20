
import pandas as pd
import numpy as np
from traderDqnAgent import Agent
import utils;
import matplotlib.pyplot as plt
from Env import Env
import collections
from typing import Deque
from traderBranchingDqn import ExperienceReplayMemory
import os
from datetime import datetime
import locale
locale.setlocale(locale.LC_NUMERIC, 'de_CH.utf8')

class Runner:
     def __init__(self):
          self.path = './runs/Output'

     def remove_model(self): 
          MODELS_PATH = "models"
          MODEL_PATH = os.path.join(MODELS_PATH, "model_state_dict") # online model 

          if os.path.exists(MODEL_PATH):
               os.remove(MODEL_PATH)

          if not os.path.exists(MODELS_PATH):
               os.makedirs(MODELS_PATH)

     def train_test_split(self, df):
          split_at = len(df) * 3//4
          train_data = df.iloc[:split_at]
          test_data = df.iloc[split_at:]
          return train_data, test_data

     def run(self, df_prices, num_episodes_train):
          train_data, test_data = self.train_test_split(df_prices)
          agent = Agent(Env(train_data))
          agent.train(num_episodes_train)

          agent = Agent(Env(test_data))
          agent.epsilon = 0
          data = agent.bid()
          print("Cumulated return is=", data[0])
          print("Count of successfull bids is=", data[1])
          print("Accuracy is=", data[1]/agent.n_successfull_bids)
          return data

     def store_params(self, **kwargs):
          # write the currently best parameters according to a given keyfigure into the file
          # overwrite the file
          with open("C:\\temp\\best_params_DQN.txt", 'w') as f:
               for kw in kwargs:
                    f.write('{}: {:.5f}\n'.format(kw, kwargs[kw]))


     def plot(self, df, learning_rates, color_map, num_episodes_train, filename):
          plot_result = []
          color_map = plt.get_cmap('cool')
          colors = [color_map(i) for i in np.linspace(0, 1, len(learning_rates))]
          dqn_agent = Agent(Env(df))
          for i, lr in enumerate(learning_rates):
               dqn_agent.learning_rate = lr
               data = dqn_agent.train(num_episodes_train)
               res_tuple = (lr, colors[i], data)
               plot_result.append(res_tuple)                       
          utils.save(plot_result, filename)
     
     def plot_dynamic(self, df, arg_name, arguments, num_episodes_train, filename, label):
          plot_result = []
          color_map = plt.get_cmap('cool')
          train_data, test_data = self.train_test_split(df)
          env = Env(train_data)
          colors = [color_map(i) for i in np.linspace(0, 1, len(arguments))]
          for i, value in enumerate(arguments):
               self.remove_model()
               keywords = { arg_name: value }
               dqn_agent = Agent(env, **keywords)
               data = dqn_agent.train(num_episodes_train)
               res_tuple = (value, colors[i], data)
               plot_result.append(res_tuple)                       
          utils.save(plot_result, filename, label)
     
     def plot_dynamic_combined(self, df, arg_name1, arg_name2, arguments1, arguments2, num_episodes_train, filename, label):
          plot_result = []
          color_map = plt.get_cmap('cool')
          train_data, test_data = self.train_test_split(df)
          env = Env(train_data)
          n = len(arguments1) * len(arguments2)
          colors = [color_map(i) for i in np.linspace(0, 1, n)]
          color_idx = 0
          for i, value1 in enumerate(arguments1):
               for j, value2 in enumerate(arguments2):
                    self.remove_model()
                    keywords = { arg_name1: value1, arg_name2: value2}
                    dqn_agent = Agent(env, **keywords)
                    data = dqn_agent.train(num_episodes_train)
                    res_tuple = (value1, value2, colors[color_idx], data)
                    plot_result.append(res_tuple)
                    color_idx +=1                   
          utils.save2(plot_result, filename, label)

     def plot_hist_ratio_bids(self, df, num_episodes):
          plt.cla()
          self.remove_model()
          train_data, test_data = self.train_test_split(df)
          agent = Agent(Env(train_data))
          agent.train(num_episodes)
          test_env = Env(test_data)
          test_agent = Agent(test_env)
          results = test_agent.bid()
          x = results[1]
          fig, axs = plt.subplots(1, 1,
                              figsize =(10, 6),
                              tight_layout = True)
          plt.xlabel("Bins")
          plt.ylabel("Count of bids with bin")
          plt.title(fontdict = { 'fontsize': 14 }, 
          label = 
          'Mean profit by hour= € ' + str(round(results[0]/len(test_data), 2)) + ', ' +
          'Mean profit by bid= € ' + str(round(results[0]/len(results[1]), 2))  + ', ' +
          'Mean profit by MW= € ' + str(round(results[0]/len(results[1])/10, 2)) + '\n ' +
          'Nbr of act. hours= ' + str(results[2]) + ', ' +
          'Nbr of act.d bids= ' + str(len(results[1])) + ', ' +
          'ratio of act. bids= ' + str(round(len(results[1])/len(test_data), 2)) + ', ' +
          'Count of episodes= ' + str(num_episodes))
          plt.grid(axis='y')
          counts, bins = np.histogram(x)
          plt.hist(bins[:-1], bins, weights=counts)
          binsticks = [round(x,1) for x in bins]
          print(binsticks)
          axs.set_xticks(binsticks)
          plt.savefig(os.path.join(self.path, "histogram_results.png"))

          counts, bins = np.histogram(x)
          plt.hist(bins[:-1], bins, weights=counts)
          binsticks = [round(x,1) for x in bins]
          axs.set_xticks(binsticks)
          plt.savefig(os.path.join(self.path, "histogram_ratio_bids_2021.png"))
     
     def run_write_results_df(self, train_env, test_env, num_episodes):
          self.remove_model()
          agent = Agent(train_env)
          cumulated = agent.train(num_episodes)

          test_agent = Agent(test_env)
          test_agent.epsilon = 0
          results = test_agent.bid(cumulated)

          n = test_env.n
          mean_waterprice = results[4]/n
          mean_profit_by_mw = results[0]/len(results[1])/10
          print("mean water price during testing:", round(mean_waterprice, 2))
          print("Total hours:", n)
          print("Nbr of activated bids:", len(results[1]))
          print("Nbr of activated hours:", results[3])
          print("Cumulated profit: ", locale.format_string('%d', round(results[0], 2), grouping=True))
          print("Mean profit by bid:", round(results[0]/len(results[1]), 2))
          print("Mean profit by MW:", round(mean_profit_by_mw, 2))
          print("Mean profit by hour:", round(results[0]/n, 2))
          print("ratio act. bids/total hours:", round(len(results[1])/n, 2))
          print("ratio act. bids/activated hours:", round(len(results[1])/results[3], 2))
          print("Mean sold MW by hour:", round(len(results[1]) * 10/results[3], 2))

          # TODO
          # print("Profit over mean water price:", round(results[4]/n, 2))
          ##################################################################################
          print("Cumulated profit with 1.2 * waterprice: ", locale.format_string('%d', round(results[5], 2), grouping=True))
          print("Mean profit by MW with 1.2 * waterprice:", round(results[5]/len(results[9])/10, 2))
          print("Mean profit by hour with 1.2 * waterprice:", round(results[5]/n, 2))

          print("Cumulated profit with 1.2 * waterprice: ", locale.format_string('%d', round(results[6], 2), grouping=True))
          print("Mean profit by MW with 1.5 * waterprice:", round(results[6]/len(results[10])/10, 2))
          print("Mean profit by hour with 1.5 * waterprice:", round(results[6]/n, 2))
          
          print("Cumulated profit with 1.7* waterprice: ", locale.format_string('%d', round(results[7], 2), grouping=True))
          print("Mean profit by MW with 1.7 * waterprice:", round(results[7]/len(results[11])/10, 2))
          print("Mean profit by hour with 1.7 * waterprice:", round(results[7]/n, 2))

          print("Cumulated profit with 2 * waterprice: ", locale.format_string('%d', round(results[8], 2), grouping=True)) 
          print("Mean profit by MW with 2 * waterprice:", round(results[8]/len(results[12])/10, 2))
          print("Mean profit by hour with 2 * waterprice:", round(results[8]/n, 2))

          result_df = pd.DataFrame({
                              'mean_water_price': [round(mean_waterprice, 2)],
                              'Total hours': [n],
                              'Nbr_activated_bids_RL': [len(results[1])],
                              'Nbr_activated_hours_RL': [results[3]],
                              'Cumulated_profit_RL': [locale.format_string('%d', round(results[0], 2), grouping=True)],
                              'Mean_profit_bid_RL': [round(results[0]/len(results[1]), 2)],
                              'Mean_profit_MW_RL': [round(mean_profit_by_mw, 2)],
                              'Mean_profit_hour_RL': [round(results[0]/n, 2)],
                              'ratio_act_bids_total_hours_RL': [round(len(results[1])/n, 2)],
                              'ratio_act_bids_activated_hours_RL':[round(len(results[1])/results[3], 2)],
                              'Sold_MW_total_RL': [ locale.format_string('%d', round(len(results[1])* 10, 2), grouping=True)],
                              'Mean_sold_MW_hour_RL': [round(len(results[1]) * 10/results[3], 2)],
                              'Cumulated_profit_1_2': [locale.format_string('%d', round(results[5], 2), grouping=True)],
                              'Mean_profit_MW_1_2': [round(results[5]/len(results[9])/10, 2)], 
                              'Mean_profit_hour_1_2': [round(results[5]/n, 2)], 
                              'Cumulated_profit_1_5': [locale.format_string('%d', round(results[6], 2), grouping=True)],
                              'Mean_profit_MW_1_5': [round(results[6]/len(results[10])/10, 2)], 
                              'Mean_profit_hour_1_5': [round(results[6]/n, 2)], 
                              'Cumulated_profit_1_7': [locale.format_string('%d', round(results[7], 2), grouping=True)],
                              'Mean_profit_MW_1_7': [round(results[7]/len(results[11])/10, 2)], 
                              'Mean_profit_hour_1_7': [round(results[7]/n, 2)], 
                              'Cumulated_profit_2': [locale.format_string('%d', round(results[8], 2), grouping=True)],
                              'Mean_profit_MW_2': [round(results[8]/len(results[12])/10, 2)], 
                              'Mean_profit_hour_2': [round(results[8]/n, 2)],
                              })

          #result_df = pd.concat([result_df, entry])
          now = datetime.today().strftime('%Y_%m_%d_%H_%M')
          result_df.to_csv("./runs/Output/results_TRE_" + now  + ".csv", sep=";")
     
     def  write_results_with_hyperparams(self, train_data, test_data, num_episodes_train, activated_bids):        
          train_data.sort_values(by='datetime_start_utc', ascending=True)
          train_start_date = train_data.iloc[0]['datetime_start_utc'] # strftime('%Y_%m_%d_%H%M')
          train_end_date = train_data.iloc[len(train_data) -1]['datetime_start_utc'] # .strftime('%Y_%m_%d_%H%M')
          
          test_data.sort_values(by='datetime_start_utc', ascending=True)
          test_start_date = test_data.iloc[0]['datetime_start_utc']# .strftime('%Y_%m_%d_%H%M')
          test_end_date = test_data.iloc[len(test_data) -1]['datetime_start_utc'] #.strftime('%Y_%m_%d_%H%M')

          result_df = pd.DataFrame(columns=['train_start_date', 'train_end_date', 'test_start_date', 'test_end_date', 'replay_buffer_size','train_start','gamma', 'epsilon_min','epsilon_decay', 'num_episodes_train', 'learning_rate', 'num_update_steps' 'mean_profit_MW', 'mean_sold_MW_hour'])
          #, 'random_cumulated_reward','best_possible_reward'])

          # tune Hyperparameters TEST
          learning_rates = [1e-4]
          epsilon_decays = [0.99, 0.95]
          batch_sizes = [256]
          gammas = [0.9, 0.85]
          replay_buffer_sizes = [50_000, 100_000]
          num_update_steps = [10_000]
         
          for replay_buffer_size in replay_buffer_sizes:
               for gamma in gammas:
                    for epsilon_decay in epsilon_decays:
                         for lr in learning_rates:
                              for batch_size in batch_sizes:
                                   for num_update_step in num_update_steps:
                                        self.remove_model()
                                        train_agent = Agent(Env(train_data, activated_bids))
                                        # add hyperparameters to train_agent
                                        train_agent.replay_buffer_size = replay_buffer_size
                                        train_agent.memory = ExperienceReplayMemory(train_agent.replay_buffer_size)
                                        train_agent.gamma = gamma # Discountfactor, to caluclate bellmann Formula
                                        train_agent.epsilon_decay = epsilon_decay
                                        train_agent.learning_rate = lr
                                        train_agent.batch_size = batch_size
                                        train_agent.num_update_steps = num_update_step
                                        train_agent.train(num_episodes_train)

                                        # create axpo_agent
                                        axpo_agent = Agent(Env(test_data, activated_bids))    
                                        results = axpo_agent.bid()
                                        mean_profit_MW = results[0]/len(results[1])/10
                                        entry = pd.DataFrame({
                                             'train_start_date': [train_start_date],
                                             'train_end_date': [train_end_date],
                                             'test_start_date': [test_start_date],
                                             'test_end_date': [test_end_date],
                                             'train_start': [replay_buffer_size],
                                             'replay_buffer_size': [replay_buffer_size],
                                             'train_start': [20_000], 
                                             'gamma': [gamma], 
                                             'epsilon_min': [0.1],
                                             'epsilon_decay': [epsilon_decay], 
                                             'num_episodes_train': [num_episodes_train], 
                                             'learning_rate': [lr], 
                                             'num_update_steps': [num_update_step],
                                             'mean_profit_MW': [mean_profit_MW],
                                             'mean_sold_MW_hour': [round(len(results[1]) * 10/results[3], 2)], 
                                             })
                                        
                                        result_df = pd.concat([result_df, entry])
          
                                                  
          result_df.to_csv("./runs/Output/hyperparametersC_TRE_bidding.csv", sep=";")
