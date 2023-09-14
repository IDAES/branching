import os
import glob
import gzip
import argparse
import pickle
import multiprocessing as mp
import shutil
import time
import sys

import numpy as np
import ecole
from branching.utilities import log

import natsort

class ExploreThenStrongBranch:
    def __init__(self, expert_probability, rng):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores(pseudo_candidates=False)
        self.rng = rng

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(self.rng.choice(np.arange(2), p=probabilities)) #returns 1 with prob self.expert_probability
        if expert_chosen:
            return (self.strong_branching_function.extract(model,done), True)
        else:
            return (self.pseudocosts_function.extract(model,done), False)

def send_orders(orders_queue, instances, seed, query_expert_prob, node_limit, out_dir):

    # This RNG determines only the (instance, seed) pair to be processed next
    rng = np.random.RandomState(seed)

    episode = 0
    while True:
        instance = rng.choice(instances) # replace = True, same instance can be selected but solved with a different seed possibly
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, query_expert_prob, node_limit, out_dir])
        episode += 1


def make_samples(in_queue, out_queue):

    while True:

        sample_counter = 0

        episode, instance, seed, query_expert_prob, node_limit, out_dir = in_queue.get()

        scip_parameters = {
                        'separating/maxrounds': 0, 'presolving/maxrestarts': 0,
                        'limits/nodes': node_limit, 'timing/clocktype': 1}

        obs_func = {
            "node_gnn_candidates": ecole.observation.NodeBipartiteCand(),
            "khalil_observation": ecole.observation.Khalil2016(),
        }

        rng_node_record = np.random.RandomState(seed)
        observation_function = { "scores": ExploreThenStrongBranch(expert_probability=query_expert_prob, rng = rng_node_record),
                                 "observations": obs_func }
        env = ecole.environment.Branching(observation_function=observation_function,
                                scip_params=scip_parameters, pseudo_candidates=False)

        print(f"[w {os.getpid()}] episode {episode}, seed {seed}, "
              f"processing instance '{instance}'...\n", end='')
        
        # Signaling start of an instance, not used for sampling
        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        # Instance is solved with its own seed
        env.seed(seed) 

        observation, action_set, _, done, _ = env.reset(instance)
        
        while not done:

            scores, scores_are_expert = observation["scores"]
            node_observation = observation["observations"]

            node_gnn_candidates = node_observation["node_gnn_candidates"]
            khalil_observation = node_observation["khalil_observation"]

            node_observation = (node_gnn_candidates.column_features[action_set,:], # use candidate features only
                                khalil_observation.features[action_set,:]) # use candidate features only                                )

            action = action_set[scores[action_set].argmax()]

            if scores_are_expert:
                data = [node_observation, action, action_set, scores[action_set]] # need only scores of the action set
                filename = f'{out_dir}/sample_{episode}_{sample_counter}.pkl' #unique identifier for sample
                
                with gzip.open(filename, 'wb') as f:
                    pickle.dump({
                        'episode': episode,
                        'instance': instance,
                        'seed': seed,
                        'data': data,
                        }, f)
                out_queue.put({
                    'type': 'sample',
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'filename': filename,
                    'ncands': len(action_set)
                })
                sample_counter += 1

            try:
                observation, action_set, _, done, _ = env.step(action)
            except Exception as e:
                done = True
                with open("error_log.txt","a") as f:
                    f.write(f"Error occurred solving {instance} with seed {seed}\n")
                    f.write(f"{e}\n")

        print(f"[w {os.getpid()}] episode {episode} done, {sample_counter} samples\n", end='')
        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def collect_samples(instances, out_dir, rng, n_cands, n_jobs,
                    query_expert_prob, node_limit):

    """
    Function to solve instances and collect samples.


    Parameters
    ----------
    instances : list of str
        List of the names of the instance files

    out_dir : str
        The location of samples

    rng : Generator
        Random number generator

    n_cands : int
        Max. number of candidate measurements
    
    n_jobs : int
        Number of parallel jobs for solving instances
    
    query_expert_prob : float
        Probability of performing strong branching at a node
    
    node_limit : int
        Limit on the number of nodes when solving an instance, to limit too much 
        data from a single instance in the dataset.


    Returns
    -------
    n_episodes : int
        Number of instances solved during sampling
    n_samples : int
        Total number of samples (nodes) collected during sampling
    total_candidates : int
        Total number of measurements (candidates) collected during sampling

    """

    os.makedirs(out_dir, exist_ok=True)

    # start workers
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher = mp.Process(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), query_expert_prob,
                  node_limit, tmp_samples_dir),
            daemon=True)
    dispatcher.start()

    # The RNG here is used only once, to create a seed for the RNG in send_orders

    workers = []
    for i in range(n_jobs):
        p = mp.Process(
                target=make_samples,
                args=(orders_queue, answers_queue),
                daemon=True)
        workers.append(p)
        p.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    n_samples = 0
    total_candidates = 0
    n_episodes = 0
    in_buffer = 0
    while total_candidates < n_cands:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode] # contains samples of type 'sample' and 'done' only
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode] # delete key-value pair from dict
                    current_episode += 1

                # else write sample
                else:
                    #print(sample['filename'])
                    sample_ncands = sample['ncands'] #len(action_set)
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    n_samples += 1
                    total_candidates += sample_ncands
                    n_episodes = sample['episode'] + 1
                    print(f"[m {os.getpid()}] {n_samples} samples written, {total_candidates} cand.s collected, "
                          f"ep {sample['episode']} ({in_buffer} in buffer).\n", end='')

                    # early stop dispatcher
                    if total_candidates >= n_cands and dispatcher.is_alive():
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...\n", end='')
                        buffer = {}
                        break

    # # stop all workers
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True) #won't raise a file not found error
    
    return [n_episodes, n_samples, total_candidates]
