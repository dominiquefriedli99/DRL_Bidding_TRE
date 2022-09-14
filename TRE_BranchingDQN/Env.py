import pandas as pd
import numpy as np
from sqlalchemy import desc
import torch
import utils

class Env:
  def __init__(self, df_states, df_activated_bids):
    self.state_features = ['spot_price_actual', 'estimated_max_price', 'estimated_min_price','estimated_mean_price','estimated_sum_volume_activated', 'estimated_std', 
    'estimated_count_bids', 'estimated_count_successfull_bids', 'weekday', 'is_working_day', 'hour', 'forecast']
    # self.keys = ['max_price', 'min_price','mean_price','sum_volume_activated', 'std', 'count_bids', 'count_successfull_bids', 'volume', 'datetime_start_utc']
    self.df_states = df_states
    self.df_activated_bids = df_activated_bids
    self.n = len(df_states)
    self.current_idx = 0
    
    self.states = self.df_states[self.state_features].to_numpy()
    #self.key_objects = self.df_states[self.keys]
    self.observation_space = self.states[self.current_idx]
    self.n_successfull_bids = len(df_states[df_states['sum_volume_activated'] > 0])
    
    # hardcoded staff
    self.count_bid = 5
    self.action_space = [0,1,2,3,4]
    self.volume_bid = 10
    # Branching parameters
    self.bins = 12
    self.low_price = 1.4
    self.high_price = 10
    self.discretized = np.array([1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 3, 4, 5, 7.5, 10]) # np.linspace(self.low_price, self.high_price, self.bins).astype(int)

  '''def getReward(self, bids):
    # first sucessfull bid = 1, 2nd successfull = 3... linear interpolation 
    reward = 0
    max_price = self.df_states.iloc[self.current_idx]['max_price']
    min_price =  self.df_states.iloc[self.current_idx]['min_price']
    if max_price == 0:
      return reward
    reward_factors = [1, 2, 4, 8, 30]
    bids = np.sort(bids)
    for i, bid in enumerate(bids):
      if bid > min_price and bid < max_price:
        reward += reward_factors[i]
      else:
        return reward
    return reward'''

  '''def getRewardByConfIntervall(self, bids):
    # first sucessfull bid = 1, 2nd successfull = 3... linear interpolation 
    total_rewards = 0
    max_price = self.df_states.iloc[self.current_idx]['max_price']
    min_price =  self.df_states.iloc[self.current_idx]['min_price']
    if max_price == 0:
      return total_rewards 
    for bid in bids:
      if bid > min_price and bid < max_price:
        total_rewards += 2 # default reward
        p_value_bid = utils.getPValue(self.key_objects.iloc[self.current_idx], bid)
        if p_value_bid > 0.9:
          # high risk taken with success
          total_rewards += 100
        elif p_value_bid > 0.8:
          # high risk taken with success
          total_rewards += 50
        elif p_value_bid < 0.1:
          total_rewards -= 1
    return total_rewards'''

  '''def getRewardByProfit(self, bids):
    reward = 0
    # first sucessfull bid = 1, 2nd successfull = 3... linear interpolation 
    max_price = self.df_states.iloc[self.current_idx]['max_price']
    min_price =  self.df_states.iloc[self.current_idx]['min_price']
    if max_price == 0:
      return reward 
    waterprice = self.df_states.iloc[self.current_idx]['spot_price_actual']
    for bid in bids:
      if bid > min_price and bid < max_price:
        profit = bid - waterprice
        profit_rate = waterprice/profit
        if profit_rate > 10:
          reward += 100
        elif profit_rate > 7:
          reward += 50
        elif profit_rate > 3:
          reward += 10
        else:
          reward += 1
    return reward'''

  '''def getBestReward(self):
    # returns the best reward. bid with highest price, which was activated minus waterprice
    max_price = self.df_states.iloc[self.current_idx]['max_price']
    if max_price == 0: 
      return 0
    current_water_price = self.df_states.iloc[self.current_idx]['spot_price_actual']
    return max_price - current_water_price'''

  # to start a new episode
  def reset(self):
    self.current_idx = 0
    state = self.states[self.current_idx]
    state = np.array(state).reshape(1, -1)
    state = torch.tensor(state).reshape(1, -1).float()
    return state
 
  def step(self, actions_idx):
    all_actions = []
    all_actions = [self.discretized[x] for x in actions_idx]
    #print("Bid is: ", self.action_space[action] * self.states[self.current_idx][0])
    reward, activated_actions = self.__calculateEffectivReturn(all_actions)
    # calculate next step
    next_state, done = self.__next_step()
    return next_state, reward, done, activated_actions, all_actions

  def getRewardForFixBehaviour(self, factor):
      actions = [factor for x in range(self.count_bid)]
      # # value = [x for x in range(len(self.discretized)) if self.discretized[x] == factor][0]
      # indx = list(self.discretized).index(factor)
      # actions = [indx for x in range(self.count_bid)]
      return self.__calculateEffectivReturn(actions)

  def getWaterPrice(self):
     return self.df_states.iloc[self.current_idx]['spot_price_actual']

  def __next_step(self):
    self.current_idx += 1
    if self.current_idx > self.n - 1:
      raise Exception("Computer says no: Episode is already finished. You shouldn't be here!")
    next_state = self.states[self.current_idx]
    next_state = np.array(next_state).reshape(1, -1)
    next_state = torch.tensor(next_state).reshape(1, -1).float()
    done = (self.current_idx == self.n - 1)
    return next_state, done

  def __calculateEffectivReturn(self, actions):
    # Calculate the effective returns according to the bids and the spot price (water price)
    successful_bids = []
    total_reward = 0.0
    count_activated_bids = self.df_states.iloc[self.current_idx]['count_successfull_bids']
    if count_activated_bids == 0:
        # no volume activated, reward is 0
        return total_reward, successful_bids
    
    max_price =  self.df_states.iloc[self.current_idx]['max_price']
    sum_volume_activated =  self.df_states.iloc[self.current_idx]['sum_volume_activated']
    current_water_price = self.df_states.iloc[self.current_idx]['spot_price_actual']
    date_time_utc = self.df_states.iloc[self.current_idx]['datetime_start_utc']

    activated_bids, activated_actions = self.__checkBidsByMeritOrder(actions, max_price, sum_volume_activated, date_time_utc)
    for bid in activated_bids:
      total_reward += (bid - current_water_price) * self.volume_bid
    return total_reward, activated_actions

  def __checkBidsByMeritOrder(self, actions, max_price, sum_volume_activated, date_time_utc):
    successfull_bids = []
    activated_actions = []

    # get Bids
    waterprice = self.df_states.iloc[self.current_idx]['spot_price_actual']
    
    activated_bids = self.df_activated_bids[self.df_activated_bids['datetime_start_utc'] == date_time_utc][['price','volume']]
    # put into merit order
    activated_bids.sort_values(by = ['price', 'volume'], axis=0, ascending=[True, True], inplace=True)
    actions = np.sort(actions)
    for action in actions:
      axpo_bid = waterprice * action #self.discretized[action]
      if axpo_bid < max_price:
        for price, volumne in zip(activated_bids['price'], activated_bids['volume']):
          if price < axpo_bid:
            sum_volume_activated -= volumne
            if sum_volume_activated <= 0:
              return successfull_bids, activated_actions
          elif axpo_bid <= price and sum_volume_activated > 0:
            successfull_bids.append(axpo_bid)
            activated_actions.append(action) # activated_actions.append(self.discretized[action])
            sum_volume_activated -=  self.volume_bid
            break
      else: 
        break
    return successfull_bids, activated_actions

 