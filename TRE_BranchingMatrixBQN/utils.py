from calendar import c
import numpy as np 
from argparse import ArgumentParser 
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d
import scipy.stats as st
from datetime import datetime, date
from workalendar.europe import Switzerland
import locale
locale.setlocale(locale.LC_NUMERIC, 'de_CH.utf8')
cal = Switzerland()

def normalize_column(values):
  min = np.min(values)
  max = np.max(values)
  norm = (values - min)/(max-min) 
  return(pd.DataFrame(norm))

def get_min_and_max(values):
  return np.min(values), np.max(values)

def add_column_and_merge(df_bids, name_column):
    new_column_name = 'estimated_' + name_column
    df = df_bids[[name_column, 'datetime_start_utc']]
    df = df.sort_values(by='datetime_start_utc', ascending=True)
    df[new_column_name] = df[name_column].shift()

    df_bids = pd.merge(df_bids, df[['datetime_start_utc', new_column_name]],   # merge previous group value back to dataframe by group 
                                on='datetime_start_utc', how='left', validate='many_to_one')
    return df_bids

def add_conf_int_t_low(grp):
    prices = grp[grp['volume_activated'] > 0][['price']].values
    return grp.assign(conf_int_t_05 = 
                      st.t.interval(alpha=0.95, df=len(prices)-1,
                      loc=np.mean(prices),
                      scale=st.sem(prices))[0][0])

def add_conf_int_t_high(grp):
    prices = grp[grp['volume_activated'] > 0][['price']].values
    return grp.assign(conf_int_t_95 = 
                      st.t.interval(alpha=0.95, df=len(prices)-1,
                      loc=np.mean(prices),
                      scale=st.sem(prices))[1][0])

def add_conf_int_normal_low(grp):
    prices = grp[grp['volume_activated'] > 0][['price']].values
    return grp.assign(conf_int_normal_05 = 
                      st.norm.interval(alpha=0.95,
                      loc=np.mean(prices),
                      scale=st.sem(prices))[0][0])

def add_conf_int_normal_high(grp):
    prices = grp[grp['volume_activated'] > 0][['price']].values
    return grp.assign(conf_int_normal_95 = 
                      st.norm.interval(alpha=0.95,
                      loc=np.mean(prices),
                      scale=st.sem(prices))[1][0])

def calc_conf_int(grp):
    prices = grp[grp['volume_activated'] > 0][['price']].values
    return grp.assign(conf_int_normal_95 = 
                      st.norm.interval(alpha=0.95,
                      loc=np.mean(prices),
                      scale=st.sem(prices))[1][0])

def add_is_working_day(row):
    return row.assign(is_working_day = int(cal.is_working_day(row.iloc[0]['datetime_start_utc'])))


def getPValue(state, x):
    # return the probability that the variables x takes on a value is less than or equal the result in a t-distribution.
    confidenceLevel = 0.95   # 95% CI given
    degrees_freedom = state['count_successfull_bids'] - 1  #degree of freedom = sample size-1
    pvalue = st.t.cdf(x, degrees_freedom, state['mean_price'],  state['std'])
    return pvalue

def save(data, filename, label):
    path = './runs/Output'
    try: 
        os.makedirs(path)
    except: 
        pass 
    values, colors, rewards, cum_profits = zip(*data)
    fig = plt.figure(figsize = (10, 5))
    plt.title('Cumulated profit on testdata with different noise values')
    plt.ylabel('Cumulated profit')
    plt.xlabel('Episodes')
    for i, reward in enumerate(rewards): 
        x = list(np.arange(1, len(rewards[0]) + 1))
        plt.plot(x, reward, color=colors[i], label= label + ' (' + str(values[i]) + ')= € ' + locale.format_string('%.2f', cum_profits[i], grouping=True))
        plt.legend(loc='best')
    plt.savefig(os.path.join(path, filename))

def save2(data, filename, label):
    path = './runs/Output'
    try: 
        os.makedirs(path)
    except: 
        pass 
    values1, values2, colors, rewards = zip(*data)
    fig = plt.figure(figsize = (10, 5))
    plt.title('Mean profit achieved in training')
    plt.ylabel('mean profit per hour')
    plt.xlabel('Episodes')
    for i, reward in enumerate(rewards): 
        x = list(np.arange(1, len(rewards[0]) + 1))
        plt.plot(x, reward, color=colors[i], label= label + '=' + f'{values1[i]:,}' + '/' +  f'{values2[i]:,}')
        plt.legend(loc='best')
    plt.savefig(os.path.join(path, filename))

def plot_eps_functions(epsilons):
    path = './runs/Output'
    try: 
        os.makedirs(path)
    except: 
        pass 
    fig = plt.figure(figsize = (10, 5))
    plt.title('Decaying epsilon function')
    plt.ylabel('value of epsilon')
    plt.xlabel('timesteps')
    plt.plot(epsilons, c='b', alpha=0.3)
    plt.savefig(os.path.join(path, "decaying_epsilons.png"))


def plot_progress(cumulated_profit_episode, axpo_total_return):
    path = './runs/Output'
    try: 
        os.makedirs(path)
    except: 
        pass 
    '''plt.cla()
    plt.plot(cumulated_reward, c = 'b', alpha = 0.3)
    plt.plot(gaussian_filter1d(cumulated_reward, sigma = 3), c = 'black', label = 'Mean profit')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulated reward in €')
    plt.title('Cumulated reward while train in €')
    plt.savefig(os.path.join(path, 'cumulated_reward.png'))'''

    # pd.DataFrame(rewards, columns = ['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index = False)
    fig = plt.figure(figsize = (9, 6))
    plt.cla()
    plt.clf()
    plt.plot(cumulated_profit_episode, c='b', alpha=0.3)
    plt.plot(gaussian_filter1d(cumulated_profit_episode, sigma=3), c='black', label='Cumulated profit in €')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulated profit while train in €')
    plt.title("current profit while train")
    if axpo_total_return is not None:
        title = "Cumulated Profit on test-data: € " + locale.format_string('%.2f', axpo_total_return, grouping=True)
        plt.title(title)
    plt.savefig(os.path.join(path, 'cumulated_profit.png'))

def plot_reward_and_profit(cumulated_reward, cumulated_profit, value_for_title):
    path = './runs/Output'
    try: 
        os.makedirs(path)
    except: 
        pass
    if len(cumulated_reward) > 0:
        plt.cla()
        plt.plot(cumulated_reward, c = 'b', alpha = 0.3)
        plt.plot(gaussian_filter1d(cumulated_reward, sigma = 3), c = 'black', label = 'Mean profit')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulated Reward in €')
        plt.title('Cumulated reward while train in €')
        if value_for_title is not None:
            title = "Cumulated Reward in € " + locale.format_string('%.2f', value_for_title, grouping=True)
            plt.title(title)
        plt.savefig(os.path.join(path, 'cumulated_reward.png'))

    # pd.DataFrame(rewards, columns = ['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index = False)
    fig = plt.figure(figsize = (9, 6))
    plt.cla()
    plt.clf()
    if len(cumulated_profit) > 0:
        plt.plot(cumulated_profit, c='b', alpha=0.3)
        plt.plot(gaussian_filter1d(cumulated_profit, sigma=3), c='black', label='Mean profit by MWh in €')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulated profit in €')
        plt.title("Cumulated profit while train in €")
        if value_for_title is not None:
            title = "Cumulated Profit in € " + locale.format_string('%.2f', value_for_title, grouping=True)
            plt.title(title)
        plt.savefig(os.path.join(path, 'cumulated_profit.png'))