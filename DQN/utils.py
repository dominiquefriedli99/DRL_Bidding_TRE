from calendar import c
import numpy as np 
from argparse import ArgumentParser 
import os
import numpy as np

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d

def save(data, filename, label):
    path = './runs/Output'
    try: 
        os.makedirs(path)
    except: 
        pass 

    values, colors, rewards = zip(*data)
    x = list(np.arange(1, len(rewards[0]) + 1))
    plt.title('training rewards by episodes')
    plt.ylabel('Cumulated reward')
    plt.xlabel('Episodes')
    for i, reward in enumerate(rewards):    
        plt.plot(x, reward, color=colors[i], label=label + str(format(values[i],'.1e' )))
        plt.legend(loc='best')
        #fig, ax = plt.subplots(figsize=(10, 5))
        #plt.cla()
        #plt.clf()
        #plt.plot(gaussian_filter1d(axpo_rewards, sigma = 3), linewidth=2, c = 'b', label = 'Axpo-bot')
        #plt.plot(rewards[i], linewidth=2, c = colors[i], label = 'lr=' + str(lrs[i]))
        
    plt.savefig(os.path.join(path, filename))
    # pd.DataFrame(axpo_rewards, columns = ['Reward']).to_csv(os.path.join(path, 'reward.csv'), index = False)

'''def save(data, filename):
    path = './runs/Output'
    try: 
        os.makedirs(path)
    except: 
        pass 

    colors, rewards, labels = zip(*data)
    x = list(np.arange(1, len(rewards[0]) + 1))
    plt.title('training rewards with different agents')
    plt.ylabel('Cumulated reward')
    plt.xlabel('Episodes')
    for i, reward in enumerate(rewards):    
        plt.plot(x, reward, color=colors[i], label=labels[i])
        plt.legend(loc='best')
        #fig, ax = plt.subplots(figsize=(10, 5))
        #plt.cla()
        #plt.clf()
        #plt.plot(gaussian_filter1d(axpo_rewards, sigma = 3), linewidth=2, c = 'b', label = 'Axpo-bot')
        #plt.plot(rewards[i], linewidth=2, c = colors[i], label = 'lr=' + str(lrs[i]))
        
    plt.savefig(os.path.join(path, filename))'''
    # pd.DataFrame(axpo_rewards, columns = ['Reward']).to_csv(os.path.join(path, 'reward.csv'), index = False)