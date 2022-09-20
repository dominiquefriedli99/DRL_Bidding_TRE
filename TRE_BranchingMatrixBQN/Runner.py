
import pandas as pd
import numpy as np
from traderDqnAgent import Agent
import utils;
import matplotlib.pyplot as plt
import seaborn as sns
from Env import Env
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

     def run(self, df_prices, df_activated_bids, num_episodes_train):
          train_data, test_data = self.train_test_split(df_prices)
          agent = Agent(Env(train_data, df_activated_bids))
          cumulated_profit = agent.train(num_episodes_train)

          agent = Agent(Env(test_data, df_activated_bids))
          agent.epsilon = 0
          data = agent.bid(cumulated_profit)
          print("Cumulated return is=", data[0])
          print("Count of successfull bids is=", data[1])
          return data

     def store_params(self, **kwargs):
          # write the currently best parameters who gives the best accuracy into the file
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
          test_env = Env(test_data)
          test_agent = Agent(test_env)
          colors = [color_map(i) for i in np.linspace(0, 1, len(arguments))]
          for i, value in enumerate(arguments):
               self.remove_model()
               keywords = { arg_name: value }
               dqn_agent = Agent(env, **keywords)
               data = dqn_agent.train(num_episodes_train)
               res_tuple = (value, colors[i], data)
               plot_result.append(res_tuple)                       
          utils.save(plot_result, filename, label)

     def tune_noise(self, train_env, test_env, arguments, num_episodes_train):
          plot_result = []
          color_map = plt.get_cmap('cool')
          colors = [color_map(i) for i in np.linspace(0, 1, len(arguments))]
          for i, value in enumerate(arguments):
               self.remove_model()
               train_env.noise = value
               data = Agent(train_env).train(num_episodes_train)
               cum_profit = Agent(test_env).bid([])
               res_tuple = (value, colors[i], data[0], cum_profit[0])
               plot_result.append(res_tuple)                       
          utils.save(plot_result, "noises3.png", "noise")
     
     def plot_dynamic_combined(self, train_env, arg_name1, arg_name2, arguments1, arguments2, num_episodes_train, filename, label):
          plot_result = []
          color_map = plt.get_cmap('cool')
          n = len(arguments1) * len(arguments2)
          colors = [color_map(i) for i in np.linspace(0, 1, n)]
          color_idx = 0
          for i, value1 in enumerate(arguments1):
               for j, value2 in enumerate(arguments2):
                    self.remove_model()
                    keywords = { arg_name1: value1, arg_name2: value2}
                    dqn_agent = Agent(train_env, **keywords)
                    data = dqn_agent.train(num_episodes_train)
                    res_tuple = (value1, value2, colors[color_idx], data[0])
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
          cumulated_profit, epsilons = agent.train(num_episodes)
          
          test_agent = Agent(test_env)
          test_agent.epsilon = 0
          results = test_agent.bid(cumulated_profit)

          # axpo_total_return, self.successfull_actions, self.all_actions, 
          # count_activated_hours, total_water_price, 
          # self.axpo_return_1_2_waterprice, self.axpo_return_1_3_waterprice, self.axpo_return_1_5_waterprice
          n = test_env.n
          cumulated_volume = np.sum([x[1] for x in results[1]])
          mean_volume = cumulated_volume/len(results[1])
          mean_waterprice = results[2]/n
          mean_profit_by_mw = results[0]/len(results[1])/mean_volume
          print("Start training: ", train_env.df_states.iloc[0]['datetime_start_utc'])
          print("End training: ", train_env.df_states.iloc[train_env.n - 1]['datetime_start_utc'])
          print("Start testing: ", test_env.df_states.iloc[0]['datetime_start_utc'])
          print("End testing: ", test_env.df_states.iloc[test_env.n - 1]['datetime_start_utc'])
          print("Mean water price: €", round(mean_waterprice, 2))
          print("Mean water price during activated hours: €", round(results[13]/results[3], 2))
          
          print("Total hours:", n)
          print("Reinforcement Learning Agent:")
          print("Nbr of activated bids:", len(results[1]))
          print("Nbr of activated hours:", results[3])
          print("Cumulated profit: €", round(results[0], 2))
          print("Mean profit by bid: €", round(results[0]/len(results[1]), 2))
          print("Mean profit by MW: €", round(mean_profit_by_mw, 2))
          print("Mean profit by hour: €", round(results[0]/n, 2))
          print("ratio act. bids/total hours:", round(len(results[1])/n, 2))
          print("ratio act. bids/activated hours:", round(len(results[1])/results[3], 2))
          print("Mean sold MW by hour: MW", round(len(results[1]) * mean_volume/results[3], 2))

          # TODO
          profit_over_waterprice = results[0]/n/results[2]/n
          # print("Profit over mean water price:", round(results[4]/n, 2))
          ##################################################################################
          print("Agent with 1.2 * waterprice:")
          cumulated_profit_1_2 = results[8]
          cumulated_volume_1_2 =  np.sum([x[1] for x in results[4]])
          print("Cumulated profit with: €", locale.format_string('%d', round(cumulated_profit_1_2, 2), grouping=True))
          print("Mean profit by MW:  €", round(cumulated_profit_1_2/np.max([len(results[4]), 1])/10, 2))
          print("Mean profit by hour: €", round(cumulated_profit_1_2/n, 2))

          print("Agent with 1.5 * waterprice:")
          cumulated_profit_1_5 = results[9]
          cumulated_volume_1_5 =  np.sum([x[1] for x in results[5]])
          print("Cumulated profit: €", locale.format_string('%d', round(cumulated_profit_1_5, 2), grouping=True))
          print("Mean profit by MW: €", round(cumulated_profit_1_5/np.max([len(results[5]), 1])/10, 2))
          print("Mean profit by hour: €:", round(cumulated_profit_1_5/n, 2))
          
          print("Agent with 1.7 * waterprice:")
          cumulated_profit_1_7 =  results[10]
          cumulated_volume_1_7 =  np.sum([x[1] for x in results[6]])
          print("Cumulated profit: €", locale.format_string('%d', round(cumulated_profit_1_7, 2), grouping=True))

          print("Mean profit by MW: €", round(cumulated_profit_1_7/np.max([len(results[6]), 1])/10, 2))
          print("Mean profit by hour: €", round(cumulated_profit_1_7/n, 2))

          print("Agent with 1.3 * waterprice:")
          cumulated_profit_1_3 = results[11]
          cumulated_volume_1_3 =  np.sum([x[1] for x in results[7]])
          print("Cumulated profit: €", locale.format_string('%d', round(cumulated_profit_1_3, 2), grouping=True)) 
          print("Mean profit by MW: €", round(cumulated_profit_1_3/np.max([len(results[7]), 1])/10, 2))
          print("Mean profit by hour: €", round(cumulated_profit_1_3/n, 2))
          mio = 1_000_000
          result_df = pd.DataFrame({
                              'Start_training': [train_env.df_states.iloc[0]['datetime_start_utc']],
                              'End_training': [train_env.df_states.iloc[train_env.n - 1]['datetime_start_utc']],
                              'Start_testing': [test_env.df_states.iloc[0]['datetime_start_utc']],
                              'End_testing': [test_env.df_states.iloc[test_env.n - 1]['datetime_start_utc']],
                              'mean_water_price': [round(mean_waterprice, 2)],
                              'mean_water_price_activated_hours': [round(results[13]/results[3], 2)],
                              'Total_hours': [n],
                              'Total_activated_hours':[ test_env.n_successfull_bids],
                              'Nbr_activated_bids_RL': [len(results[1])],
                              'Nbr_activated_hours_RL': [results[3]],
                              'Cumulated_profit_RL': [ round(results[0], 2)],
                              'Mean_profit_by_bid_RL': [round(results[0]/len(results[1]), 2)],
                              'Mean_profit_by_MW_RL': [round(mean_profit_by_mw, 2)],
                              'Mean_profit_by_hour_RL': [round(results[0]/n, 2)],
                              'Ratio_waterprice_bid_price': [round(results[0]/n/mean_waterprice, 2)],
                              'Ratio_waterprice_activated_hours_mean_bid_price': [round(results[0]/len(results[1]/results[13]/results[3]), 2)],
                              'Ratio_act_hours_total_hours_RL': [round(results[3]/n, 2)],
                              'Ratio_act_bids_activated_hours_RL':[round(len(results[1])/results[3], 2)],
                              'Cumulated_MW_RL': [round(cumulated_volume, 2)],
                              'Mean_sold_MW_activated_hour_RL': [round(cumulated_volume/results[3], 2)],
                              'Index_RL': [round(results[0] * mean_profit_by_mw /mio, 2)],
                              'Cumulated_profit_1_2': [round(cumulated_profit_1_2, 2)],
                              'Mean_profit_by_MW_1_2': [round(cumulated_profit_1_2/cumulated_volume_1_2, 2)],
                              'Mean_profit_by_bid_1_2': [round(cumulated_profit_1_2/len(results[4]), 2)], 
                              'Mean_profit_by_hour_1_2': [round(cumulated_profit_1_2/n, 2)],
                              'Index_1_2': [round(cumulated_profit_1_2 * cumulated_profit_1_2/cumulated_volume_1_2/mio, 2)],
                              'Cumulated_profit_1_3': [round(cumulated_profit_1_3, 2)],
                              'Mean_profit_by_MW_1_3': [round(cumulated_profit_1_3/np.max([cumulated_volume_1_3, 1]), 2)], 
                              'Mean_profit_by_bid_1_3': [round(cumulated_profit_1_3/np.max([len(results[7]), 1]), 2)],
                              'Mean_profit_by_hour_1_3': [round(cumulated_profit_1_3/n, 2)],
                              'Index_1_3': [round(cumulated_profit_1_3 * cumulated_profit_1_3/cumulated_volume_1_3/mio, 2)],
                              'Cumulated_profit_1_5': [round(cumulated_profit_1_5, 2)],
                              'Mean_profit_by_MW_1_5': [round(cumulated_profit_1_5/cumulated_volume_1_5, 2)],
                              'Mean_profit_by_bid_1_5': [round(cumulated_profit_1_5/len(results[5]), 2)], 
                              'Mean_profit_by_hour_1_5': [round(cumulated_profit_1_5/n, 2)],
                              'Index_1_5': [round(cumulated_profit_1_5 * cumulated_profit_1_5/cumulated_volume_1_5/mio, 2)],
                              })
          
          result_df.to_csv("./runs/Output/results.csv", sep=";")

          successfull_bidded_volume = [x[1] for x in results[1]]
          successfull_bidded_price_ratios = [x[2] for x in results[1]]
          all_bidded_volume = [x[1] for x in results[12]]
          all_bidded_price_ratios = [x[0] for x in results[12]]
          # scatterplot
          df = pd.DataFrame(data=results[1], columns=['price', 'volume', 'price ratio'])
          fig = plt.figure(figsize = (9, 6))
          plt.cla()
          plt.clf()
          plt.title("Scatterplot price volume")
          sns.scatterplot(data=df, x="price", y="volume")
          sns.set_theme(style="whitegrid")
          sns.set(rc={"figure.figsize":(9, 6)})
          plt.savefig('./runs/Output/scatterplot_price_volume.png')

          x = successfull_bidded_volume
          df = pd.DataFrame(x, columns=['a'])
          df['a'] = df['a'].astype(float)
          df['b'] = 1
          ax = df.groupby('a').count().astype(int).plot(kind='bar')
          ax.set_ylabel('Count of bids')
          ax.set_xlabel('Volume in MWh')
          _ = ax.set_title('Volume of successfull bids')
          sns.set_theme(style="whitegrid")
          sns.set(rc={"figure.figsize":(9, 6)})
          plt.savefig('./runs/Output/volume_successfull_bids.png')

          x = successfull_bidded_price_ratios
          df = pd.DataFrame(x, columns=['a'])
          df['a'] = df['a'].astype(float)
          df['b'] = 1
          ax = df.groupby('a').count().astype(int).plot(kind='bar')
          ax.set_ylabel('Count of bids')
          ax.set_xlabel('Ratio of waterprice')
          _ = ax.set_title('Price ratio successfull bids')
          sns.set_theme(style="whitegrid")
          sns.set(rc={"figure.figsize":(9, 6)})
          plt.savefig('./runs/Output/price_successfull_bids.png')

          x = all_bidded_price_ratios
          df = pd.DataFrame(x, columns=['a'])
          df['a'] = df['a'].astype(float)
          df['b'] = 1
          ax = df.groupby('a').count().astype(int).plot(kind='bar')
          ax.set_ylabel('Count of bids')
          ax.set_xlabel('Ratio of waterprice')
          _ = ax.set_title('Price ratio of all proposed bids')
          sns.set_theme(style="whitegrid")
          sns.set(rc={"figure.figsize":(9, 6)})
          plt.savefig('./runs/Output/price_ratio_all_bids.png')

          x = all_bidded_volume
          df = pd.DataFrame(x, columns=['a'])
          df['a'] = df['a'].astype(float)
          df['b'] = 1
          ax = df.groupby('a').count().astype(int).plot(kind='bar')
          ax.set_ylabel('Count of bids')
          ax.set_xlabel('Volume in MWh')
          _ = ax.set_title('Volume of all proposed bids')
          sns.set_theme(style="whitegrid")
          sns.set(rc={"figure.figsize":(9, 6)})
          plt.savefig('./runs/Output/volume_all_bids.png')

          utils.plot_eps_functions(epsilons)
     
     def  write_results_with_hyperparams(self, train_data, test_data, activated_bids, num_episodes_train):        
          train_data.sort_values(by='datetime_start_utc', ascending=True)
          train_start_date = train_data.iloc[0]['datetime_start_utc'] # strftime('%Y_%m_%d_%H%M')
          train_end_date = train_data.iloc[len(train_data) -1]['datetime_start_utc'] # .strftime('%Y_%m_%d_%H%M')
          
          test_data.sort_values(by='datetime_start_utc', ascending=True)
          test_start_date = test_data.iloc[0]['datetime_start_utc']# .strftime('%Y_%m_%d_%H%M')
          test_end_date = test_data.iloc[len(test_data) -1]['datetime_start_utc'] #.strftime('%Y_%m_%d_%H%M')

          result_df = pd.DataFrame()

          # tune Hyperparameters TEST
          learning_rates = [1e-4, 1e-5]
          batch_sizes = [128, 256]
          gammas = [0.85, 0.9]
          replay_buffer_sizes = [25_000]
          num_update_steps = [5_000, 10_000]
          learning_starts = [5_000, 10_000]
          noises = [5, 11, 25, 100, 1000]   
          for batch_size in  batch_sizes:
               for replay_buffer_size in replay_buffer_sizes:
                    for gamma in gammas:
                         for lr in learning_rates:
                              for l_start in learning_starts:
                                   for num_update_step in num_update_steps:
                                        self.remove_model()
                                        train_env = Env(train_data, activated_bids)
                                        train_agent = Agent(train_env)
                                        # add hyperparameters to train_agent
                                        train_agent.replay_buffer_size = replay_buffer_size
                                        train_agent.memory = ExperienceReplayMemory(train_agent.replay_buffer_size)
                                        train_agent.gamma = gamma
                                        # train_agent.epsilon_decay = epsilon_decay
                                        train_agent.learning_rate = lr
                                        train_agent.batch_size = batch_size
                                        train_agent.num_update_steps = num_update_step
                                        train_agent.learning_start = l_start
                                        train_agent.train(num_episodes_train)
                                        # create axpo_agent
                                        axpo_agent = Agent(Env(test_data, activated_bids))    
                                        results = axpo_agent.bid([])
                                        cumulated_profit = results[0]
                                        cumulated_volume = np.sum([x[1] for x in results[1]])
                                        mean_profit_by_mw = cumulated_profit/cumulated_volume
                                        index = (cumulated_volume * mean_profit_by_mw)/1_000_000
                                        entry = pd.DataFrame({
                                                  'train_start_date': [train_start_date],
                                                  'train_end_date': [train_end_date],
                                                  'test_start_date': [test_start_date],
                                                  'test_end_date': [test_end_date],
                                                  'num_episodes': [num_episodes_train],
                                                  'batch_size' : [batch_size],
                                                  'learning_rate': [lr],
                                                  'replay_buffer_size': [replay_buffer_size], 
                                                  'learning_start': [l_start], 
                                                  'gamma': [gamma], 
                                                  'num_update_step': [num_update_step],  
                                                  'train_start': [l_start], 
                                                  'mean_proft_by_mwh': [round(mean_profit_by_mw, 2)],
                                                  'cumulated_proft': [round(cumulated_profit, 2)],
                                                  "index": [index]
                                                  })
                                             
                                        result_df = pd.concat([result_df, entry])
                                             
               result_df.to_csv("./runs/Output/results_hpyertuning_matrix_V4.csv", sep=";")

     def  hyper_run(self, train_env, test_env, year, times):
          train_agent = Agent(train_env) 
          test_agent = Agent(test_env)
          result_df = pd.DataFrame()
          for i in range(times):
               self.remove_model()
               train_agent = Agent(train_env) 
               cumulated_profit, epsilons = train_agent.train(12)
               
               test_agent = Agent(test_env)
               results = test_agent.bid(cumulated_profit)

               cumulated_profit = results[0]
               cumulated_volume = np.sum([x[1] for x in results[1]])
               mean_volume = cumulated_volume/len(results[1])
               mean_profit_by_mw = results[0]/len(results[1])/mean_volume
               index = mean_profit_by_mw * cumulated_profit /1_000_000
               print("idx=", index)
               
               entry = pd.DataFrame({
                         'test_run': [i],
                         'cumulated_profit': [cumulated_profit],
                         'cumulated_volume': [cumulated_volume],
                         'mean_profit_by_mw': [mean_profit_by_mw],
                         'index': [index],
                         })
                                             
               result_df = pd.concat([result_df, entry])                           
          result_df.to_csv("./runs/Output/results_hyperrun_" + str(year) + ".csv", sep=";")
