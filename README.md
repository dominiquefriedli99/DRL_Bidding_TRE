# Introduction
This project "TRE_MaxtrixBidding" is the final implementaton. 

It is based on the implementation of a PyTorch version of Branching Double Deep Q-Learning from repo https://github.com/MoMe36/BranchingDQN.
The TRE_MaxtrixBidding implements a Reinforcement Learning-Agent who optimizes 5 prices for bidding for Swiss TRE market in up direction.
The action Space contains discretized vector for volume (like 5, 7, 10... 50) used for volume (in MWh) and discretized vector for price ratios between 1.3 and 2.5 used for price (price ratio * da-ahead-price) in € for the bids.

The merit oder principle is implemented in the reward function: Bids are odered by price then by volume, volume is activated until sum of volume activated is reached and until the maximal volume of 50 MWh is reached. This is a contraint of the energy trader for this specific market.
The implemented agent (traderDqnAgent) uses the bidding data from Swissgrid of the last hour, the day-ahead price and forecast data of consumption/production for the current hour to estimate the bids.

the data is in Folder/Data. There is no data for the moment, as the files were to large.

In the Dataframe.py the dataset is featured to add state information for the agent.

The count of episodes to train can be set in the _init_.py file. 15 epsisodes is suggested. 
The bid-method runs the trained model for the test data and returns cumulated profit, successfull bids and control results for fix behaviour (like always 1.2 * day-ahead-price) and much more key figures and informations.

the Project "DQN" contains a prototype of the bidding strategy, implememtation is based on https://github.com/mome36/DoubleDQN

the Project "TRE_BranchingDQN" is based on the implementation of a PyTorch version of Branching Double Deep Q-Learning from repo https://github.com/MoMe36/BranchingDQN. It optimizes the price for 5 bids only.

# Getting Started
The final implementation is implemented with Anaconda3 2022.11
# DRL_Bidding_TRE
Realtime Bidding strategy for Swiss TRE-market with DRL algorithm
# Introduction
The final implementation of the real-time bidding strategy for the Swiss TRE-market is in the project "TRE_MaxtrixBidding".
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

- the folder "DQN contains a prototype for the bidding strategy.

# Getting Started
The project is implemented with Anaconda3 2022.11
installed libraries: 
pytorch 1.10.2
torch 1.10.2
gym 0.23.1
pandas 1.4.2
numpy 1.22.3

# Build and Test
start the _init_.py file
on the Runner class there are different run methods to generate output or plots.
In agent's "train"-method iterates x times through the train_data and trains the model. The model will be stored in \models.
The agent's "bid"-method runs the test-data. 
Use the functions on Runner to plot train effort.

