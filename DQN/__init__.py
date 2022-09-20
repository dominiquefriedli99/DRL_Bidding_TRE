
import pandas as pd
import numpy as np
from Env import Env
from traderDqnAgent import Agent
import sql_data
from datetime import datetime
import utils;
import matplotlib.pyplot as plt
from Runner import Runner
import os


MODELS_PATH = "models"
MODEL_PATH = os.path.join(MODELS_PATH, "tre_bidder.h5") # online model 

# if os.path.exists(MODEL_PATH):
#     os.remove(MODEL_PATH)

# if not os.path.exists(MODELS_PATH):
#     os.makedirs(MODELS_PATH)

# today = datetime.today().strftime('%Y-%m-%d')
# todo: takes from environ variables
# df_bids = sql_data.pull_data_from_sql("tre_bid",  "2020-01-01", today, user, password, server, database)
df_bids = pd.read_parquet('Data/tre_bids_sample.parquet.gzip', engine='pyarrow')
df_bids.dropna(axis=0, how='all', inplace=True)
df_bids.dropna(axis=1, how='any', inplace=True)

# filter for up direction only
df_bids = df_bids[df_bids['AUCTION_MRID'] =='TREnergie+_s']

# add sum_volume_activated, max_price, min_price, mean_price
df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(sum_volume_activated=grp['volume_activated'].sum()))
df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(max_price=grp[grp['volume_activated'] > 0]['price'].max()))
df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(min_price=grp[grp['volume_activated'] > 0]['price'].min()))
df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(mean_price=grp[grp['volume_activated'] > 0]['price'].mean()))


# keep one per datetime_start_utc
df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).nth(0)
df_bids = df_bids.sort_values(by='datetime_start_utc', ascending=True)

# add new column 'estimated_volume_activated' 
df_temp = df_bids[['datetime_start_utc','sum_volume_activated']]
df_temp = df_temp.sort_values(by='datetime_start_utc', ascending=True)
df_temp['estimated_volume_activated'] = df_temp['sum_volume_activated'].shift()
df_bids = pd.merge(df_bids, df_temp[['datetime_start_utc','estimated_volume_activated']],   
                                on='datetime_start_utc', how='left', validate='many_to_one')

# add column estimated max price
df_temp2 = df_bids[['max_price', 'datetime_start_utc']]
df_temp2 = df_temp2.sort_values(by='datetime_start_utc', ascending=True)
df_temp2['estimated_max_price'] = df_temp2['max_price'].shift()

df_bids = pd.merge(df_bids, df_temp2[['datetime_start_utc','estimated_max_price']],   # merge previous group value back to dataframe by group 
                                on='datetime_start_utc', how='left', validate='many_to_one')

# add column estimted min price
df_temp3 = df_bids[['min_price', 'datetime_start_utc']]
df_temp3 = df_temp3.sort_values(by='datetime_start_utc', ascending=True)
df_temp3['estimated_min_price'] = df_temp3['min_price'].shift()

df_bids = pd.merge(df_bids, df_temp3[['datetime_start_utc','estimated_min_price']],   # merge previous group value back to dataframe by group 
                                on='datetime_start_utc', how='left', validate='many_to_one')

# estimted mean price
df_temp4 = df_bids[['mean_price', 'datetime_start_utc']]
df_temp4 = df_temp4.sort_values(by='datetime_start_utc', ascending=True)
df_temp4['estimated_mean_price'] = df_temp4['mean_price'].shift()

df_bids = pd.merge(df_bids, df_temp4[['datetime_start_utc','estimated_mean_price']],   # merge previous group value back to dataframe by group 
                                on='datetime_start_utc', how='left', validate='many_to_one')
                  
df_bids.fillna(0, inplace=True) # put 0 instead Nan
df_bids = df_bids[1:] # delete first row as Nan in estimated_volume

# cast to datetime for merge 
df_bids['datetime_start_utc'] = pd.to_datetime(df_bids['datetime_start_utc'])

# get Spot-Prices from csv
day_ahead_prices = pd.read_csv('Data/ch_spot_price_actual.csv')

# read from DB
# day_ahead_prices.drop('Price_area', axis=1, inplace=True)
day_ahead_prices['datetime_start_utc'] = pd.to_datetime(day_ahead_prices['datetime_start_utc'])
day_ahead_prices = day_ahead_prices.sort_values(by='datetime_start_utc', ascending=True)
# rename as from SQL, columns is named spot_price_actual
day_ahead_prices.rename({'spot_price_ch': 'spot_price_actual'}, axis=1, inplace=True)

# merge the dataframes by datetime_start_utc
df_bids_prices = day_ahead_prices.merge(df_bids, on='datetime_start_utc')
df_bids_prices.to_csv("C:\\temp\\df_bids.csv")

#df_bids_prices = pd.read_csv("C:\\temp\\df_bids.csv")
num_episodes_train = 30
num_episodes_test = 1

n = len(df_bids_prices)
split_at = n * 3//4
train_data = df_bids_prices.iloc[:split_at]
test_data = df_bids_prices.iloc[split_at:]
print(len(test_data))

# cmap = plt.get_cmap('cool')
# runner = Runner()
# labels = ['RL agent', '1.2 * waterprice', '1.5 * waterprice', '2 * waterprice']
# #runner.plot(train_data, epsilon_decays, cmap, 1, 'eps_decays.png')
# #runner.plot_dynamic(train_data, "learning_rate", learning_rates, cmap, num_episodes_train, 'learning_rates_2.png', 'lr=')
# #runner.plot_dynamic_agents(train_data, num_episodes_train, cmap, labels, 'diff_agents_static_behaviour_20_episodes.png')
# # now run the agent with test-data. The epsilon will be set to 0, means: always take greedy action never select the action randomly
# runner.plot_results(train_data, test_data, num_episodes_train)

train_agent = Agent(Env(train_data))
train_agent.train(num_episodes_train)

agent = Agent(Env(test_data))
agent.epsilon = 0
result = agent.bid(num_episodes_test)

print("result axpo_total_reward: ", result[0])
print("result best cumulated reward: ", result[1])
print("result take_1_2_waterprice_reward: ", result[2])
print("result take_1_5_waterprice_reward: ", result[3])
print("result take_2_waterprice_reward: ", result[4])
print("nbr activated hours", result[5])

#return (round(axpo_total_reward, 2), round(insider_total_reward, round, 2), (take_1_2_waterprice_reward, 2), round(take_1_5_waterprice_reward, 2), round(take_2_waterprice_reward, 2), count_reward)


