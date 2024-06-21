import argparse
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
import yaml

from tqdm import tqdm

import utils
from agent import Agent
from environment import Environment

def run_reversal_learning(ensemble_count, reversal_count, trial_count, tau_val, k_val, k_decay, position=0):
    success_rate = np.array([0] * (trial_count * (reversal_count+1)))

    disable_tqdm = position != 0
    for ensemble_idx in tqdm(range(ensemble_count), disable=disable_tqdm):
        environment = Environment()

        agent = Agent(tau_val=tau_val, k_val=k_val)
        agent.current_action = np.random.choice(environment.action_list)

        while True:
            environment._step(agent)

            if (environment.current_timestep % trial_count) == 0:
                if environment.rewarded_action == 'A':
                    environment.rewarded_action = 'B'
                else:
                    environment.rewarded_action = 'A'

                agent.k_val *= (1-k_decay)

            if environment.current_timestep == (trial_count * (reversal_count+1)):
                break

        success_rate += (np.array(environment.rewarded_action_list) == np.array(agent.memory_actions)).astype(int)

    if ensemble_count == 1:
        action_list, stimulus_list, memory_list, p_list = agent.memory_actions, agent.memory_tendency_stimulus, agent.memory_tendency_memory_A, agent.memory_p_A

        run_output = {
            'action_list': action_list,
            'stimulus_list': stimulus_list,
            'memory_list': memory_list,
            'p_list': p_list,
            'run_stat': success_rate,
        }
    else:
        run_stat = success_rate / ensemble_count

        run_output = {'run_stat': run_stat}

    return run_output

def calibrate_reversal_learning(run_target, config, stage=0, stage2_param=None):
    # Environment parameters
    ensemble_count = config['ENSEMBLE_COUNT']
    reversal_count = config['REVERSAL_COUNT']
    trial_count = config['TRIAL_COUNT']

    # ABC parameters
    if stage in [0, 1]:
        k_min = config['K_MIN']
        k_max = config['K_MAX']
        tau_min = config['TAU_MIN']
        tau_max = config['TAU_MAX']

        if stage == 1:
            reversal_count = 1
    elif stage == 2:
        decay_min = config['DECAY_MIN']
        decay_max = config['DECAY_MAX']
        reversal_count
    else:
        raise ValueError('Stage not support; stage can only be 0, 1, 2')

    abc_error_threshold = config['ABC_ERROR_THRESHOLD']
    distance_type = config['DISTANCE_TYPE']

    # Sample from prior until distance is smaller than threshold
    distance = abc_error_threshold + 1
    while distance > abc_error_threshold:
        # Set seed so that each mp process has a different set of random numbers
        seed = os.getpid() + np.random.randint(0, 2**31)
        np.random.seed(seed)

        if stage in [0, 1]:
            k_val = np.random.uniform(k_min, k_max)
            tau_val = np.random.uniform(tau_min, tau_max)
            k_decay = 0
        elif stage == 2:
            k_val = stage2_param['k_val']
            tau_val = stage2_param['tau_val']
            k_decay = np.random.uniform(decay_min, decay_max)

        # Run simulation
        run_output = run_reversal_learning(ensemble_count, reversal_count, trial_count, tau_val, k_val, k_decay, position=1)

        distance = utils.calculate_distance(run_target, run_output['run_stat'], distance_type)

    out = {
        'tau_val': tau_val,
        'k_val': k_val,
        'k_decay': k_decay,
        'distance': distance,
        'run': run_output['run_stat'],
    }

    return out

def calibrate_reversal_learning_star(args):
    """
    Helper function to run calibrate_reversal_learning with multiple arguments
    """
    return calibrate_reversal_learning(*args)

def run_abc(run_target, config, stage=0, stage2_param=None):
    """
    Runs the ABC algorithm across multiple processes
    run_config is a dictionary containing the parameters for the run
    """

    # Get sample_cnt from run_config
    abc_sample_count = config['ABC_SAMPLE_COUNT']

    # Run ABC in parallel
    with mp.Pool(mp.cpu_count()-2) as pool:
        args = list(zip([run_target] * abc_sample_count, [config] * abc_sample_count, [stage] * abc_sample_count, [stage2_param] * abc_sample_count))
        results = list(tqdm(pool.imap(calibrate_reversal_learning_star, args), total=abc_sample_count))

    return results

def main(config):

    # Name of experiment
    experiment_type = config['EXPERIMENT_TYPE']

    '''
    This part of the code is for setting up where to load the parameters and store the output of the simulation runs.
    '''
    config, output_folder, records_file, df_records = utils.initialize_experiment(config)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    '''
    This part of the code is to run the simulation.
    '''
    # Run simulation
    if experiment_type == 'run':

        seed = np.random.randint(0, 2**32 - 1)  # 1999154595 (seed number for Figure 3)
        np.random.seed(seed)

        ensemble_count = config['ENSEMBLE_COUNT']
        reversal_count = config['REVERSAL_COUNT']
        trial_count = config['TRIAL_COUNT']
        tau_val = config['TAU_VAL']
        k_val = config['K_VAL']
        k_decay = config['K_DECAY']

        run_output = run_reversal_learning(ensemble_count, reversal_count, trial_count, tau_val, k_val, k_decay)

        with open(f'{output_folder}/results.pkl', 'wb') as f:
            pickle.dump(run_output, f)

    elif experiment_type == 'calibrate':

        # Environment parameters
        simulation_type = config['SIMULATION_TYPE']
        target_type = config['TARGET_TYPE']

        # Get target data
        run_target = utils.get_target(target_type)

        reversal_count = config['REVERSAL_COUNT']
        trial_count = config['TRIAL_COUNT']
        run_target = run_target[:(trial_count * (reversal_count+1))]

        if simulation_type == 'single':
            results_abc = run_abc(run_target, config)

            with open(f'{output_folder}/results.pkl', 'wb') as f:
                pickle.dump(results_abc, f)
        elif simulation_type == 'serial':
            # First stage
            target_single = run_target[:(trial_count * 2)]

            results_abc_stage1 = run_abc(target_single, config, stage=1)

            with open(f'{output_folder}/results_stage1.pkl', 'wb') as f:
                pickle.dump(results_abc_stage1, f)

            # Second stage
            tau_k_pairs = []
            for run_no in results_abc_stage1:
                tau_k_pairs.append([run_no['tau_val'], run_no['k_val']])

            map_estimate = utils.get_MAP(tau_k_pairs)
            map_estimate = {'tau_val': map_estimate[0],
                            'k_val': map_estimate[1]}

            results_abc_stage2 = run_abc(run_target, config, stage=2, stage2_param=map_estimate)

            with open(f'{output_folder}/results_stage2.pkl', 'wb') as f:
                pickle.dump(results_abc_stage2, f)

    # Store config file
    with open(f'{output_folder}/config.yaml', 'w') as configfile:
        yaml.dump(config, configfile, sort_keys=False)

    # Update records file
    df_records = pd.concat([df_records, pd.DataFrame.from_dict(config, orient='index').T], ignore_index=True)
    df_records.to_csv(f'{records_file}', index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Name of config file')
    args = parser.parse_args()

    with open(f'config/{args.config}') as configfile:
        config = yaml.safe_load(configfile)

    main(config)