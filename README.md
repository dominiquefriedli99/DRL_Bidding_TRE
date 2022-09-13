# DRL_Bidding_TRE
Realtime Bidding strategy for Swiss TRE-market with DRL algorithm
# Introduction
The final implementation of the real-time bidding strategy for the Swiss TRE-market is called "TRE_MaxtrixBidding".
It's based on the implementation of a PyTorch version of Branching Double Deep Q-Learning from repo https://github.com/MoMe36/BranchingDQN.
The TRE_MaxtrixBidding implements a Reinforcement Learning control method to evaluate 5 prices for bidding for positiv TRE market with discretized vector for volume (like 5, 7, 10... 50Mw) and discretized vector of price ratio between 1.3 and 10 used for volume and price (price ratio * waterprice) for the bids.
The merit oder princinple is implemented: Bids are odered by price then by volume, volume is activated until sum of volume activated is reached and until maxVolume is reached, we use for TRE (for the moment 50MWh).
The implemented agent (traderDqnAgent) uses the data from Swissgrid of the last hour, the day-ahead price and forecast data of consumption/production for the current hour to estimate the bids.

Constraint: maximal 50MW per hour can be sold. 
the data is in Folder /Data. Unzip the tre_data_202x first.
In the Dataframe.py the dataset is featured to add state information for the agent.
The project implements the training of the model with neuronal networks and the DQN-control method. 
The count of episodes to train can be set in the _init_.py file. 15 epsisodes is suggested. 
The bid-method runs the trained model for the test data and returns cumulated profit, successfull bids and control results for fix behaviour (like always 1.2 * waterprice).

# Getting Started
The project is implemented with Anaconda3 2022.11
installed libraries: 
pytorch 1.10.2
torch 1.10.2
gym 0.23.1
pandas 1.4.2
numpy 1.22.3
