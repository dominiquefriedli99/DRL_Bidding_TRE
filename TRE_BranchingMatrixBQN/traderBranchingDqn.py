import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 
import random
from torchvision import models

class DuelingNetwork(nn.Module): 

    def __init__(self, obs, ac): 

        super().__init__()

        self.model = nn.Sequential(nn.Linear(obs, 128), 
                                   nn.ReLU(), 
                                   nn.Linear(128,128), 
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_head = nn.Linear(128, ac)
        summary(self.model , (obs, 128, 128), 32)

    def forward(self, x): 

        out = self.model(x)

        value = self.value_head(out)
        adv = self.adv_head(out)

        q_val = value + adv - adv.mean(1).reshape(-1,1)
        return q_val


class BranchingQNetwork(nn.Module):

    def __init__(self, obs, ac_dim, n): 

        super().__init__()

        self.ac_dim = ac_dim # anzahl haupt-actions LunarLanderContinuous-v2 - 2
        self.n = n # bins

        self.model = nn.Sequential(nn.Linear(obs, 128), 
                                   nn.ReLU(),
                                   nn.Linear(128,128), 
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(128, n) for i in range(ac_dim)])

        # summary(self.adv_heads, (128, n), 64)

    def forward(self, x): 

        out = self.model(x)
        value = self.value_head(out) # state value
        advs = torch.stack([l(out) for l in self.adv_heads], dim = 1)


        # print(advs.shape)
        # print(advs.mean(2).shape)
        test =  advs.mean(2, keepdim = True)
        # input(test.shape)
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim = True )
        # input(q_val.shape)

        return q_val

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for b in batch: 
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])


        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)