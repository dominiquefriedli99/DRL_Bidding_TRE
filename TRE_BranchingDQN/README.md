# Introduction
This implementation of TRE_Bidding is based on the implementation of a PyTorch version of Branching Double Deep Q-Learning from repo https://github.com/MoMe36/BranchingDQN.
The TRE_BranchingDQN implements a Reinforcement Learning control method to evaluate 5 prices for bidding (10 MWh each) for the positive TRE market (direction up).
The calculateEffectivReturn functions returns the effetive return for the 5 bids with the formula: Sum (b1 - b5) (current water price - bid ) * 10
The implemented agent (traderDqnAgent) uses the data from Swissgrid of the last hour and the day-ahead price for the current hour to estimate the bids which should be hopefully activated by Swissgrid with a profit for Axpo.
the traindata are in Folder Data: tre_bids_sample.parquet and ch_spot_price_actual.csv
In the _init_.py file there is some feature engineering implemented to enrich the environment with more states to evalate the current bids.
The project implements the training of the model with neuronal networks and the DQN-control method. 
The count of episodes to train can be set in the _init_.py file.
The bid-method runs the trained model with the calculated weights hours and summarizes the revenue done as a validation.

# Getting Started
The project is implemented with Anaconda3 2022.11
installed libraries: 
pytorch 1.10.2
torch 1.10.2
gym 0.23.1
pandas 1.4.2
numpy 1.22.3

1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
start the _init_.py file
The agent will train x times (run the train_data) and then run the "bid"-method for the train_data. 
Use the functions on Runner to plot train effort
The bid-method prints out:
Total cumulated reward with TRE Bidding optimization: the cumulated revenue with the trained model
Total reward just random: the cumulated revenue acting just randomly
Best reward possible with insider knowledge: the cumulated revenue, always with the best bid out from the data of Swissgrid

# Contribute
If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)