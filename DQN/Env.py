import pandas as pd
import numpy as np


class Env:
  def __init__(self, df_states):
    self.feats = ['spot_price_actual', 'estimated_max_price', 'estimated_min_price','estimated_mean_price','estimated_volume_activated']
    self.df_states = df_states
    self.n = len(df_states)
    self.current_idx = 0
    self.action_space = [1.01, 1.05, 1.1, 1.2, 1.3, 1.5, 1.7, 1.8, 2.0, 3.0, 5.0, 7.5, 10.0]
    self.states = self.df_states[self.feats].to_numpy()
    self.observation_space = self.states[self.current_idx]
    self.n_successfull_bids = len(df_states[df_states['sum_volume_activated'] > 0])

  def getReward(self, action):
    reward = 0.
    max_price = self.df_states.iloc[self.current_idx]['max_price']
    min_price =  self.df_states.iloc[self.current_idx]['min_price']
    if max_price > 0:
      axpo_bid = self.action_space[action] * self.states[self.current_idx][0]
      if axpo_bid >= min_price and axpo_bid <= max_price:
        #print("the bid was inside the successfull range.")
        return self.calculateReward(axpo_bid)
      else: return reward
    else: return reward

  def getRewardForFixBehaviour(self, multiplier):
    reward = 0.
    max_price = self.df_states.iloc[self.current_idx]['max_price']
    min_price =  self.df_states.iloc[self.current_idx]['min_price']
    if max_price > 0:
      axpo_bid = multiplier * self.states[self.current_idx][0]
      if axpo_bid >= min_price and axpo_bid <= max_price:
        #print("the bid was inside the successfull range.")
        return self.calculateReward(axpo_bid)
      else: return reward
    else: return reward
    

  def calculateReward(self, axpo_bid):
    # Calculate the effective reward according spot price (water price) of given index and our bid: can be positiv or negativ
    # todo: vor oder nach increment??
    current_water_price = self.df_states.iloc[self.current_idx]['spot_price_actual']
    return axpo_bid - current_water_price

  def getBestReward(self):
    # returns the best reward. bid with highest price, which was activated minus waterprice
    max_price = self.df_states.iloc[self.current_idx]['max_price']
    if max_price == 0: 
      return 0
    current_water_price = self.df_states.iloc[self.current_idx]['spot_price_actual']
    return max_price - current_water_price

  # to start a new episode
  def reset(self):
    self.current_idx = 0
    # self.current_hour = self.start_date

    # back to first item
    state = self.states[self.current_idx]
    return state
 
  def step(self, action):
    #print("Bid is: ", self.action_space[action] * self.states[self.current_idx][0])
    # the environment has compute the reward: the profit from bid minus current price of water
    reward = self.getReward(action)
    self.current_idx += 1
    if self.current_idx > self.n - 1:
      raise Exception("Computer says no: Episode is already finished. You shouldn't e here!")
    next_state = self.states[self.current_idx]
    done = (self.current_idx == self.n - 1)
    #if done: 
      #print("Episode done!")
    return next_state, reward, done

 