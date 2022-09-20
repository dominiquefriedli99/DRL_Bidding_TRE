import pandas as pd
import numpy as np
from sqlalchemy import desc
import torch
import utils

class Env:
  def __init__(self, df_states, df_activated_bids):
    self.state_features = ['spot_price_actual', 'estimated_max_price', 'estimated_min_price','estimated_mean_price', 'estimated_median_price', 'estimated_sum_volume_activated', 'estimated_std', 
    'estimated_count_successfull_bids', 'weekday', 'is_working_day', 'hour', 'forecast']
    
    self.df_states = df_states
    self.df_activated_bids = df_activated_bids
    #self.n = len(df_states) - 1
    self.current_idx = 0
    for name in self.df_states[self.state_features].columns:
      if name == 'spot_price_actual':
          #self.df_states['spot_price_actual_diff'] = np.log(self.df_states['spot_price_actual']).diff()
          self.df_states['spot_price_actual_log'] = np.log(self.df_states['spot_price_actual']).diff()
            # self.df_states['spot_price_actual_diff'] = utils.normalize_column(self.df_states['spot_price_actual']) #np.log(self.df_states['spot_price_actual']).diff()
      elif name in ['estimated_max_price', 'estimated_mean_price', 'estimated_min_price', 'estimated_mean_price','estimated_median_price']:
          self.df_states[name] =  np.log(self.df_states[name]).diff()
    self.df_states = self.df_states[1:]
    self.df_states.fillna(0, inplace=True)
    self.df_states.replace([np.inf, -np.inf], 0, inplace=True)
    #add spot_price_actual_diff to the features and remove spot_price_actual
    self.state_features.insert(0, 'spot_price_actual_log')
    self.state_features.remove('spot_price_actual')
    self.states = self.df_states[self.state_features]
    #self.states.to_csv("Data/states.csv", sep=";")
    self.states = self.df_states[self.state_features].to_numpy()
    self.observation_space = self.states[self.current_idx]
    self.n = len(self.df_states)
    self.n_successfull_bids = len(df_states[df_states['sum_volume_activated'] > 0])
    
    self.noise = None # result after hypertuning
    
    # Branching parameters, hardcoded staff
    self.count_bid = 5
    self.action_space = [0,1,2,3,4]
    self.maxVolume = 50
    self.discretizedPrices = np.array([1.3, 1.33, 1.35, 1.38, 1.4, 1.5, 1.8, 2, 5]) # price ratios 
    #self.discretizedPrices = np.array([1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 5, 7, 10]) old price rations
    self.discretizedVolumes = np.array([5, 7, 10, 15, 20, 25, 30, 50])
    self.bins = len(self.discretizedPrices) * len(self.discretizedVolumes)
    self.matrix = []
    for i in self.discretizedPrices:
      for j in self.discretizedVolumes:
        self.matrix.append([i,j])

  # to start a new episode
  def reset(self):
    self.current_idx = 0
    state = self.states[self.current_idx]
    state = np.array(state).reshape(1, -1)
    state = torch.tensor(state).reshape(1, -1).float()
    return state
 
  def step(self, actions, play: bool = False):
    bids = self.getBids(actions)
    profit, successfull_bids, reward = self.calculateProfit(bids)
    
    # calculate next step
    next_state, done = self.__next_step(play)
    return next_state, reward, done, profit, successfull_bids

  def getRewardForFixBehaviour(self, factor):
      actions = [factor for x in range(self.count_bid)]
      bids = []
      # create fixed bids with factor = price and 10MW volumw
      waterprice = self.df_states.iloc[self.current_idx]['spot_price_actual']
      for a in actions:
        bids.append([waterprice * a, 50])
      return self.calculateProfit(bids)

  def getWaterPrice(self):
     return self.df_states.iloc[self.current_idx]['spot_price_actual']
    
  def getBids(self, actions):
    bids = []
    waterprice = self.df_states.iloc[self.current_idx]['spot_price_actual']
    for action in actions:
      entry = self.matrix[action]
      price = round(waterprice * entry[0], 2)
      volume = entry[1]
      if volume > 0:
        bid = [price, volume, entry[0]]
        bids.append(bid)
    return bids

  def __next_step(self, play: bool):
    self.current_idx += 1
    if self.current_idx > self.n - 1:
      raise Exception("Computer says no: Episode is already finished. You shouldn't be here!")
    
    next_state = self.states[self.current_idx]
    if not play and self.noise is not None and self.current_idx % self.noise == 0:
      next_state = self.get_noise(next_state)
    next_state = np.array(next_state).reshape(1, -1)
    next_state = torch.tensor(next_state).reshape(1, -1).float()
    done = (self.current_idx == self.n - 1)
    return next_state, done

  def calculateProfit(self, bids):
    # Calculate the effective returns according to the bids and the spot price (water price)
    successfull_bids = []
    total_profit = 0.0
    reward = 0.0
    #count_activated_bids = self.df_states.iloc[self.current_idx]['count_successfull_bids']
    count_activated_bids = self.df_states.iloc[self.current_idx]['count_successfull_bids']
    if count_activated_bids == 0:
        # no volume activated, reward is 0
        return total_profit, successfull_bids, 0

    successfull_bids = self.__checkBidsByMeritOrderAndMaxVolume(bids)
    
    #  Calculate reward
    # different reward-singals
    # all bids with volume 0: punishment of -100
    '''if sum_volume_bids == 0:
      return 0, successfull_bids, -10'''

    # volume activated but our bids are not accepted. Punishment of -100
    '''if len(successfull_bids) == 0:
      return 0, successfull_bids, -100'''

    current_water_price = self.df_states.iloc[self.current_idx]['spot_price_actual']
    for bid in successfull_bids:
        total_profit += (bid[0] - current_water_price) * bid[1]

    # reward = total_profit
    # return profit over MWh as reward, needs more episodes.
    # reward is cumulated profit * mean profit by mwh.

    '''if total_profit > 0:
        reward = total_profit * total_profit/np.sum([x[1] for x in successfull_bids])'''
    # TODO: Reward for total_bidded_volume == 50
    reward = total_profit
    return total_profit, successfull_bids, reward

  def __checkBidsByMeritOrderAndMaxVolume(self, bids):
    successfull_bids = []
    # max_price =  self.df_states.iloc[self.current_idx]['max_price']
    max_price =  self.df_states.iloc[self.current_idx]['max_price']
    swissgrid_sum_volume_activated =  self.df_states.iloc[self.current_idx]['sum_volume_activated']
    date_time_utc = self.df_states.iloc[self.current_idx]['datetime_start_utc']
    activated_bids = self.df_activated_bids[self.df_activated_bids['datetime_start_utc'] == date_time_utc][['price','volume']]
 
    # put into merit order
    activated_bids.sort_values(by = ['price', 'volume'], axis=0, ascending=[True, True], inplace=True)
    
    # sort by price, than by volume
    bids.sort()
    used_volume = 0
    for bid in bids:
        # Abbruchbedingung
      if swissgrid_sum_volume_activated <= 0 or used_volume >= self.maxVolume: 
          return successfull_bids
      if bid[0] <= max_price:
        for price, volume in zip(activated_bids['price'], activated_bids['volume']):
          if price < bid[0]:
            swissgrid_sum_volume_activated -= volume
            # sum_volume reached
            if swissgrid_sum_volume_activated <= 0:
              return successfull_bids
          if bid[0] <= price and swissgrid_sum_volume_activated > 0:
            if bid[0] == price:
              # lower volume is activated first
              if bid[1] > volume:
                swissgrid_sum_volume_activated -=  volume
                # take next bid in merit order
                continue
            
            # split volume if used volume would be higher than max volume
            effectiv_volume_activated_bid = np.min([(self.maxVolume - used_volume), bid[1]])
            effectiv_volume_activated_bid = bid[1]
            
            # maybe split the bid: volume must be not more than sum_volume_activated
            effectiv_volume_activated_bid = np.min([effectiv_volume_activated_bid, swissgrid_sum_volume_activated])         
            
            bid[1] = effectiv_volume_activated_bid
            used_volume += effectiv_volume_activated_bid
            swissgrid_sum_volume_activated -=  effectiv_volume_activated_bid
            successfull_bids.append(bid)
            break
      else: 
        break
    return successfull_bids

  def get_noise(self, state):
      state = state + np.random.normal(size=len(state))
      for i, value in enumerate(state):
        name = self.state_features[i]
        min = np.min(self.df_states[name])
        max = np.max(self.df_states[name])
        value = np.clip(value, min, max)
      return state