# Introduction 
DQN implements a Reinforcement Learning control method to evaluate the "best" price for one bid for the given hour. 
The implemented agent (traderDqnAgent) uses the data from Swissgrid of the last hour and the day-ahead price for the current hour to estimate a bid which should be hopefully activated by Swissgrid with a profit for Axpo.
In the _init_.py file there is some feature engineering implemented to enrich the environment with more states to evalate the current bid.
The project implements the training of the model with neuronal networks and the DQN-control method. 
The count of episodes to train can be set in the _init_.py file.
The bid-method runs the trained model with the calculated weights for about 1000 hours and cumulates the profit as validation.

# Getting Started
The project is implemented with Anaconda3 2022.11 with Python 3.9.7 64-bit
installed libraries: 
tensorflow 2.6.0
keras 2.6.0
tqdm 4.63.0

1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
start the _init_.py file
The agent will train x times (run the train_data) and then run the "bid"-method for the train_data.
The train- and testdata is splitted into about 2/3 and 1/3 of the dataset. 
The bid-method prints out:
Total reward with TRE Bidding optimization: the cumulated revenue with the trained model
Total reward just random: the cumulated revenue acting just randomly
Best reward possible with insider knowledge: the cumulated revenue, always with the best bid out from the data of Swissgrid

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)