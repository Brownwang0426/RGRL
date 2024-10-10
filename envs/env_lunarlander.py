
import gym

import numpy as np
import math
from scipy.special import softmax

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, TensorDataset, Subset

import csv

import multiprocessing as mp
import os
import sys
import copy
import random
import gc
import time
from tqdm import tqdm

"""
# Function for vectorizing
Crucial function regarding how you manipulate or shape your state, action and reward

- It's essential to choose between immediate rewards and summed rewards for training your agent. 
  If the current state doesn't encapsulate all crucial past information, using immediate rewards is advisable. 
  This approach prevents confusion caused by varying summed rewards for the same state.

- As for reward shaping, it is recommended to increase your reward upper and decrease your reward lower bound.
"""

def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center
    if angle_targ > 0.4: angle_targ = 0.4    # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5])*1.0
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo*20 - 1, -angle_todo*20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
        elif angle_todo < -0.05: a = 3
        elif angle_todo > +0.05: a = 1
    return a

def quantifying(array_size, min_value, max_value, value):
    array    = np.zeros(array_size) 
    interval = (max_value - min_value) / array_size
    index    = int( (value - min_value) // interval + 1)
    if index >= 0:
        array[ : index] = 1
    return array

def vectorizing_state(state):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    state_0 = quantifying(100, -1.5 , 1.5  , state[0]) 
    state_1 = quantifying(100, -1.5 , 1.5  , state[1]) 
    state_2 = quantifying(100, -5   , 5    , state[2]) 
    state_3 = quantifying(100, -5   , 5    , state[3]) 
    state_4 = quantifying(100, -3.14, 3.14 , state[4])   
    state_5 = quantifying(100, -5   , 5    , state[5])   
    state_6 = quantifying(100, 0    , 1    , state[6])    
    state_7 = quantifying(100, 0    , 1    , state[7])    
    state   = np.concatenate((state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7))   
    return state

def vectorizing_action(pre_activated_actions, training_episode, env, s):  # Reminder: change this for your specific task ⚠️⚠️⚠️
    action_size      = pre_activated_actions.size(2)
    action_argmax    = int(torch.argmax(pre_activated_actions[0, 0]))

    if (training_episode+1) <= 1:
        action_argmax = heuristic(env, s)

    return np.eye(action_size)[action_argmax], action_argmax

def vectorizing_reward(state, reward, summed_reward, done, reward_size):       # Reminder: change this for your specific task ⚠️⚠️⚠️
    reward = quantifying(reward_size, -200, 325, reward)       
    return reward


