import numpy as np
import pickle
import gzip
import argparse
import os
import json
import pickle
import logging
import itertools

from datetime import datetime
from pathlib import Path

def geo_mean_overflow(arr, shift=1):
    prod = np.exp(np.mean(np.log(arr + shift), axis=0))
    return prod - shift

def calculate_mean(my_list, type):
    my_list = np.array(my_list)
    if type == 'A':
        return np.mean(my_list)
    elif type == 'G':
        return geo_mean_overflow(my_list, shift=1)
    else:
        raise Exception

def log(str, logfile=None):
    str = f'[{datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def log2(str, logfile=None): ## Logging without date and time information
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def load_flat_samples(filename):
    with gzip.open(filename, 'rb') as file:
        sample = pickle.load(file)

    state, best_cand, cands, cand_scores = sample['data']
    
    cands = np.array(cands)
    cand_scores = np.array(cand_scores)

    assert(all(cand_scores >= 0))

    cand_states = []
    node_obs_cand = state[0]
    khalil = state[1]

    khalil -= khalil.min(axis=0, keepdims=True)
    max_val = khalil.max(axis=0, keepdims=True)
    max_val[max_val == 0] = 1
    khalil /= max_val

    norm_factor = np.sqrt(sum(np.square(cand_scores[cand_scores >= 0])))
    if norm_factor <= 0:
        norm_factor = 1
    
    score = cand_scores / norm_factor

    cand_states = np.concatenate((node_obs_cand, khalil), axis=1)

    best_cand_idx = np.where(cands == best_cand)[0][0]

    return cand_states, score, best_cand_idx
