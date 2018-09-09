# -*- coding: utf-8 -*-

import pandas as pd
import math
import random

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10

no_of_rewards_1 = [0] * d
no_of_rewards_0 = [0] * d

total_reward = 0

for n in range(0, N):
    ad = 0
    max_val = 0
    for i in range(0, d):
        random_draw = random.betavariate(no_of_rewards_1[i] + 1, no_of_rewards_0[i] + 1)
        if(random_draw > max_val):
            max_val = random_draw
            ad = i
        
    reward = dataset.values[n, ad]
    total_reward += reward
    
    if(reward > 0):
        no_of_rewards_1[ad] += 1
    else:
        no_of_rewards_0[ad] += 1
    
    
        