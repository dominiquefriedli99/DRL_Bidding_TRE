
from this import d
from tkinter.ttk import Separator
import pandas as pd
import numpy as np
from Env import Env
from traderDqnAgent import Agent
from datetime import datetime
import matplotlib.pyplot as plt
from Runner import Runner
from Dataframe import Dataframe
import utils
import os
from datetime import datetime, date
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import locale
locale.setlocale(locale.LC_NUMERIC, 'de_CH.utf8')

'''
data = Dataframe()
df_prices = data.init_df_prices()
activated_bids = data.init_df_activated_bids()
'''

df_prices = pd.read_csv("Data/df_bids_and_spot_prices_forecast_2021.csv", sep=";")
activated_bids = pd.read_csv("Data/All_activated_bids_2021.csv", sep=";")

df_prices.sort_values(by='datetime_start_utc', ascending=True, inplace=True)

split_at = len(df_prices) * 3//4
train_data= df_prices.iloc[:split_at]
test_data = df_prices.iloc[split_at:]
num_episodes_train = 15
times = 100
runner = Runner()
year = 2021
#runner.run_write_results_df(Env(train_data, activated_bids), Env(test_data, activated_bids), num_episodes_train)
#runner.write_results_with_hyperparams(train_data, test_data, activated_bids, num_episodes_train)
runner.hyper_run(Env(train_data, activated_bids), Env(test_data, activated_bids), year, 100)