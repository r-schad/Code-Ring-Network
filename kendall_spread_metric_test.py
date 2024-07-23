import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import norm, kendalltau
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime as dt
import os

from utilities import exponential, bimodal_exponential_noise
# from Code_Ring_Network import CodeRingNetwork

def clockwise_mono_spread_metric(code_activity, **metric_kwargs):
    n = len(code_activity)
    if code_activity.shape == (n,1):
        code_activity = code_activity.flatten()
    num_high = np.count_nonzero(code_activity > metric_kwargs['min_activity_value'])
    high_idxs = np.argwhere(code_activity > metric_kwargs['min_activity_value']).flatten()
    ideal = np.arange(num_high, 0, -1) # values don't matter, just the order

    # store all scores for each i,j pair, where each pair has a chain forward
    scores = np.zeros((n, n))

    # iterate over left endpoint of chain
    for i in high_idxs:
        # iterate over right endpoints of chain
        for j in high_idxs:
            print(i,j)
            if i > j:
                spread = n - i + j + 1
                # get all indexes i to n-1, then 0 to j, going fwd
                fwd_idxs_all = np.concatenate((np.arange(i, n), np.arange(0, j+1)))
            else:
                spread = j - i + 1
                # get all indexes i to j going fwd
                fwd_idxs_all = np.arange(i, j + 1)
            # get the indexes of those indexes (therefore, temp indexes) that have a high value
            fwd_high_idxs_temp = np.argwhere(code_activity[fwd_idxs_all] > metric_kwargs['min_activity_value']).flatten()
            # and go back to getting the original index value from those temp indexes
            fwd_high_idxs = fwd_idxs_all[fwd_high_idxs_temp]

            # check that our list of high indexes going forward contains all high values. if not, we skip this chain
            if num_high == len(fwd_high_idxs):
                # spread score based on distance going forward
                dist_from_perf_spread = np.abs(spread - num_high)
                print('dist_from_perf_spread: ', dist_from_perf_spread)
                top = exponential(dist_from_perf_spread, rate=metric_kwargs['spread_penalty_rate'], init_val=1)
                bottom = 1 + (metric_kwargs['weight_diff_from_desired']  * np.abs(metric_kwargs['num_desired_high'] - num_high))
                spread_score = top / bottom
                print('spread_score: ', spread_score)

                # kendall score based on chain going forward - since clockwise only, need to normalize kendall to [0,1]
                kendall = (kendalltau(code_activity[fwd_high_idxs], ideal).statistic + 1) / 2
                print('kendall: ', kendall)

                # weighted average of the two scores
                scores[i][j] = (metric_kwargs['theta'] * spread_score) + ((1 - metric_kwargs['theta']) * kendall)
    
    i, j = np.unravel_index(np.argmax(scores), shape=(n,n))
    max_score = scores[i][j]

    return max_score


# def bidir_mono_spread_metric(code_activity, **metric_kwargs):
#     n = len(code_activity)
#     if code_activity.shape == (n,1):
#         code_activity = code_activity.flatten()
#     num_high = np.count_nonzero(code_activity > metric_kwargs['min_activity_value'])
#     high_idxs = np.argwhere(code_activity > metric_kwargs['min_activity_value']).flatten()
#     ideal = np.arange(1, num_high + 1) # values don't matter, just the order

#     # store all scores for each i,j pair, where each pair has a chain either forward or backward
#     scores = np.zeros((n, n, 2))

#     # iterate over left endpoint of chain
#     for idx_of_idxs, i in enumerate(high_idxs):
#         # iterate over right endpoints of chain
#         for j in high_idxs[idx_of_idxs+1:]:
#             # print(i,j)
#             # get all indexes i to j going fwd
#             fwd_idxs_all = np.arange(i, j + 1)
#             # get the indexes of those indexes (therefore, temp indexes) that have a high value
#             fwd_high_idxs_temp = np.argwhere(code_activity[fwd_idxs_all] > metric_kwargs['min_activity_value']).flatten()
#             # and go back to getting the original index value from those temp indexes
#             fwd_high_idxs = fwd_idxs_all[fwd_high_idxs_temp]

#             # get all indexes from i to j going bkwd
#             bkwd_idxs_all = np.concatenate((np.arange(i,-1,-1), np.arange(n-1,j-1,-1)))
#             # get the indexes of those indexes (therefore, temp indexes) that have a high value        
#             bkwd_high_idxs_temp = np.argwhere(code_activity[bkwd_idxs_all] > metric_kwargs['min_activity_value']).flatten()
#             # and go back to getting the original index value from those temp indexes
#             bkwd_high_idxs = bkwd_idxs_all[bkwd_high_idxs_temp]

#             # check that our list of high indexes going forward contains all high values. if not, we skip this chain
#             if num_high == len(fwd_high_idxs):
#                 # spread score based on distance going forward
#                 spread = np.abs(i - j) + 1
#                 dist_from_perf_spread = np.abs(spread - num_high)
#                 # print('dist_from_perf_spread: ', dist_from_perf_spread)
#                 top = exponential(dist_from_perf_spread, rate=metric_kwargs['spread_penalty_rate'], init_val=1)
#                 bottom = 1 + (metric_kwargs['weight_diff_from_desired']  * np.abs(metric_kwargs['num_desired_high'] - num_high))
#                 spread_score = top / bottom

#                 # kendall score based on chain going forward
#                 kendall = np.abs(kendalltau(code_activity[fwd_high_idxs], ideal).statistic)
#                 # print('kendall: ', kendall)

#                 # weighted average of the two scores
#                 scores[i][j][0] = (metric_kwargs['theta'] * spread_score) + ((1 - metric_kwargs['theta']) * kendall)
                
#             if num_high == len(bkwd_high_idxs): # TODO: elif?
#                 # spread score based on distance going forward
#                 spread = n - np.abs(i - j) + 1
#                 dist_from_perf_spread = np.abs(spread - num_high)
#                 # print('dist_from_perf_spread: ', dist_from_perf_spread)
#                 top = exponential(dist_from_perf_spread, rate=metric_kwargs['spread_penalty_rate'], init_val=1)
#                 bottom = 1 + (metric_kwargs['weight_diff_from_desired']  * np.abs(metric_kwargs['num_desired_high'] - num_high))
#                 spread_score = top / bottom

#                 # kendall score based on chain going forward
#                 kendall = np.abs(kendalltau(code_activity[bkwd_high_idxs], ideal).statistic)
#                 # print('kendall: ', kendall)

#                 # weighted average of the two scores
#                 scores[i][j][1] = (metric_kwargs['theta'] * spread_score) + ((1 - metric_kwargs['theta']) * kendall)
    
#     i, j, dir = np.unravel_index(np.argmax(scores), shape=(n,n,2))
#     max_score = scores[i][j][dir]

#     return max_score

min_activity_value = 0.05
spread_penalty_rate = -0.5
weight_diff_from_desired = 0.2
theta = 0.35

ring_neurons = 36
num_desired_high = 9

num_high = num_desired_high

# c = np.array([3,2,1,0,0,0,0,5,0,4])
# score = clockwise_mono_spread_metric(c, 
#             min_activity_value=min_activity_value,
#             spread_penalty_rate=spread_penalty_rate,
#             weight_diff_from_desired=weight_diff_from_desired,
#             theta=theta,
#             num_desired_high=num_desired_high)
# print('Score: ', score)
# pass
f,axs = plt.subplots(12,4)
for i in range(12):
    for j in range(4):
        # c = bimodal_exponential_noise(3*ring_neurons//4,1*ring_neurons//4,6)
        # c = np.where(c > min_activity_value, c, 0.0)
        c = np.concatenate((np.linspace(0.1,1.0,num_desired_high), np.zeros(ring_neurons-num_desired_high)))
        r = np.random.randint(0, ring_neurons)
        c = np.roll(c, r)
        flip = np.random.randint(0,2)
        if flip: c = np.flip(c)
    
        # score = bidir_mono_spread_metric(c, 
        #                 min_activity_value=min_activity_value,
        #                 spread_penalty_rate=spread_penalty_rate,
        #                 weight_diff_from_desired=weight_diff_from_desired,
        #                 theta=theta,
        #                 num_desired_high=num_desired_high)
        score = clockwise_mono_spread_metric(c, 
                        min_activity_value=min_activity_value,
                        spread_penalty_rate=spread_penalty_rate,
                        weight_diff_from_desired=weight_diff_from_desired,
                        theta=theta,
                        num_desired_high=num_desired_high)

        axs[i][j].matshow(c.reshape(1,len(c)), vmin=0, vmax=1)
        axs[i][j].set_xlabel(np.round(score,3))
        plt.axis('off')

plt.tight_layout()
plt.show()
pass
