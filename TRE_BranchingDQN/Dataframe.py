
from tkinter.ttk import Separator
import pandas as pd
import numpy as np
from Env import Env
from traderDqnAgent import Agent
import sql_data
from datetime import datetime
import matplotlib.pyplot as plt
from Runner import Runner
import utils
import os
from datetime import datetime, date

class Dataframe:
    def __init__(self):
        # always re-init the model when getting data
        MODELS_PATH = "models"
        MODEL_PATH = os.path.join(MODELS_PATH, "model_state_dict") # online model 

        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)

    def init_df_prices(self):
        # df_bids = sql_data.pull_data_from_sql("tre_bid",  "2020-01-01", today, user, password, server, database)
        #df_bids = pd.read_parquet('Data/tre_bids_sample.parquet.gzip', engine='pyarrow')
        df_bids = pd.read_csv('Data/bids_2021.csv', sep=";")
        df_bids.dropna(axis=0, how='all', inplace=True)
        df_bids.dropna(axis=1, how='any', inplace=True)

        # filter for up direction only
        df_bids = df_bids[df_bids['AUCTION_MRID'] =='TREnergie+_s']

        # add count_bids, count_successfull_bids
        df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(count_bids=len(grp)))
        df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(count_successfull_bids=len(grp[grp['volume_activated'] > 0])))

        # add sum_volume_activated, max_price, min_price, mean_price, std
        df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(sum_volume_activated=grp['volume_activated'].sum()))
        df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(max_price=grp[grp['volume_activated'] > 0]['price'].max()))
        df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(min_price=grp[grp['volume_activated'] > 0]['price'].min()))
        df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(mean_price=grp[grp['volume_activated'] > 0]['price'].mean()))
        df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(std=grp[grp['volume_activated'] > 0]['price'].std()))

        # # keep one per datetime_start_utc
        df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).nth(0)
        df_bids = df_bids.sort_values(by='datetime_start_utc', ascending=True)

        # # add columns into next ts with suffix 'estimated_'
        df_bids = utils.add_column_and_merge(df_bids, 'sum_volume_activated')
        df_bids = utils.add_column_and_merge(df_bids, 'max_price')
        df_bids = utils.add_column_and_merge(df_bids, 'min_price')
        df_bids = utils.add_column_and_merge(df_bids, 'mean_price')
        df_bids = utils.add_column_and_merge(df_bids, 'std')
        df_bids = utils.add_column_and_merge(df_bids, 'count_bids')
        df_bids = utils.add_column_and_merge(df_bids, 'count_successfull_bids')

        # add weekday into df
        # cast to datetime for merge 
        df_bids['datetime_start_utc'] = pd.to_datetime(df_bids['datetime_start_utc'])
        df_bids['weekday'] = df_bids['datetime_start_utc'].dt.dayofweek
        df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: utils.add_is_working_day(grp))
        df_bids['hour'] = df_bids.datetime_start_utc.dt.hour.astype(int)
        # df_bids = utils.add_column_and_merge(df_bids, 'conf_int_t_05')
        # df_bids = utils.add_column_and_merge(df_bids, 'conf_int_t_95')
        # df_bids = utils.add_column_and_merge(df_bids, 'conf_int_normal_05')
        # df_bids = utils.add_column_and_merge(df_bids, 'conf_int_normal_95')

        df_bids.fillna(0, inplace=True) # put 0 instead Nan
        df_bids = df_bids[1:] # delete first row as Nan in estimated_volume

        # get Spot-Prices from csv
        # day_ahead_prices = pd.read_csv('Data/ch_spot_price_actual.csv')
        day_ahead_prices = pd.read_csv('Data/day_ahed_prices_2021.csv', sep=";")
        # if read from DB
        #day_ahead_prices.drop('Price_area', axis=1, inplace=True)
        day_ahead_prices['datetime_start_utc'] = pd.to_datetime(day_ahead_prices['datetime_start_utc'])
        day_ahead_prices = day_ahead_prices.sort_values(by='datetime_start_utc', ascending=True)
        # rename as from SQL, columns is named spot_price_actual
        day_ahead_prices.rename({'spot_price_ch': 'spot_price_actual'}, axis=1, inplace=True)

        # merge the dataframes by datetime_start_utc
        df_bids_prices = day_ahead_prices.merge(df_bids, on='datetime_start_utc')

        # add forecast data
        df_forecast = pd.read_csv('Data/ch_production_consumption_forecast.csv')
        df_forecast['datetime'] = pd.to_datetime(df_forecast['datetime'])
        df_forecast.rename({'datetime': 'datetime_start_utc'}, axis=1, inplace=True)
        df_forecast = df_forecast.sort_values(by='datetime_start_utc', ascending=True)
        df_forecast['forecast'] = df_forecast.apply(lambda row: row.total_production_forecast - row.consumption_forecast, axis=1)
        df_bids_prices_forecast = df_forecast.merge(df_bids_prices, on='datetime_start_utc')
        df_bids_prices_forecast = df_bids_prices_forecast[[
            'spot_price_actual', 'estimated_max_price', 'estimated_min_price','estimated_mean_price','estimated_sum_volume_activated', 'estimated_std', 
            'estimated_count_bids', 'estimated_count_successfull_bids', 'weekday', 'is_working_day', 'hour',
            'max_price', 'min_price','mean_price','sum_volume_activated', 'std', 'count_bids', 'count_successfull_bids', 'volume', 'datetime_start_utc',
            'forecast']]

          # store as CSV
        df_bids_prices_forecast.to_csv("C:\\temp\\df_bids_and_spot_prices_forecast_2021.csv", sep=";")
        return df_bids_prices_forecast
    
    def init_df_activated_bids(self):
        # df_bids = sql_data.pull_data_from_sql("tre_bid",  "2020-01-01", today, user, password, server, database)
        #df_bids = pd.read_parquet('Data/tre_bids_sample.parquet.gzip', engine='pyarrow')
        df_bids = pd.read_csv('Data/bids_2021.csv', sep=";")
        df_bids.dropna(axis=0, how='all', inplace=True)
        df_bids.dropna(axis=1, how='any', inplace=True)

        # filter for up direction only
        df_bids = df_bids[df_bids['AUCTION_MRID'] =='TREnergie+_s']
        df_bids = df_bids.groupby(["datetime_start_utc"], as_index=False).apply(lambda grp: grp.assign(sum_volume_activated=grp['volume_activated'].sum()))
        df_bids.to_csv("C:\\temp\\df_all_activated_bids_2021.csv", sep=";")
        return df_bids

    
