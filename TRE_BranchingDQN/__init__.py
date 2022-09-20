
from this import d
from tkinter.ttk import Separator
import pandas as pd
import numpy as np
from Env import Env
from traderDqnAgent import Agent
import sql_data
from datetime import datetime
import matplotlib.pyplot as plt
from Runner import Runner
from Dataframe import Dataframe
import utils
import os
from datetime import datetime, date
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''data = Dataframe()
activated_bids = data.init_df_activated_bids()'''

df_bids_prices = pd.read_csv("C:\\temp\\df_bids_and_spot_prices_forecast_2021.csv", sep=";")
activated_bids = pd.read_csv("C:\\temp\\df_all_activated_bids_2021.csv", sep=";")
df_bids_prices = df_bids_prices.sort_values(by='datetime_start_utc', ascending=True)

num_episodes_train = 12
split_at = len(df_bids_prices) * 3//4
train_data = df_bids_prices.iloc[:split_at]
test_data = df_bids_prices.iloc[split_at:]

'''
train_agent =  Agent(Env(train_data, activated_bids))
train_agent.train(num_episodes_train)

test_env = Env(test_data, activated_bids)
agent = Agent(test_env)
results = agent.bid()
n = test_env.n

mean_profit_by_mw = results[0]/len(results[1])/10
print("Mean profit by bid:", round(results[0]/len(results[1]), 2))
print("Mean profit by MW:", round(mean_profit_by_mw, 2))

print("Mean profit by MW with 1.5 * waterprice:", round(results[5]/len(results[8])/10, 2))
print("Mean profit by hour with 1.5 * waterprice:", round(results[5]/n, 2))
    
print("Mean profit by MW with 1.7 * waterprice:", round(results[6]/len(results[9])/10, 2))
print("Mean profit by hour with 1.7 * waterprice:", round(results[6]/n, 2))
    
print("Mean profit by MW with 2 * waterprice:", round(results[7]/len(results[10])/10, 2))
print("Mean profit by hour with 2 * waterprice:", round(results[7]/n, 2))
'''
runner = Runner()

# runner.write_results_with_hyperparams(train_data, test_data, num_episodes_train, activated_bids)
runner.run_write_results_df(Env(train_data, activated_bids), Env(test_data, activated_bids), num_episodes_train)

#runner.write_hist_ratio_bids(df_bids_prices, num_episodes_train)
epsilondecays = [0.95, 0.9, 0.85]
mins = [0.1, 0.01]
# runner.plot_dynamic_combined(df_bids_prices, "epsilon_decay", "epsilon_min", epsilondecays, mins, num_episodes_train, 'decaying_epsilons.png', 'eps_decay/eps_min')
#runner.plot_dynamic(df_bids_prices, "num_update_steps", num_update_steps, num_episodes_train, 'num_update_steps_hyperparapters.png', 'num_upd_step')
# now run the agent with test-data. The epsilon will be set to 0, means: always take greedy action never select the action randomly
#runner.plot_hist_ratio_bids(df_bids_prices, num_episodes_train)'''
