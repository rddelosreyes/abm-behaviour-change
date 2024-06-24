import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from scipy.stats import gaussian_kde
from scipy.optimize import minimize

from datetime import datetime
from matplotlib import rc
from sklearn.metrics import mean_squared_error, mean_absolute_error

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

DATA_DIR = "../data"

def function_logistic(x: int, x0: int=0, k: int=1, L: int=1) -> int:
    """Returns the output of a logistic function.

    Args:
        x: Any real number.
        x0: The x value of the sigmoid midpoint.
        k: Steepness of the curve.
        L: Maximum value of the curve.
    """
    return L / (1 + np.exp(-k * (x - x0)))

def function_memory_influence(memory_tendency_stimulus: list[int], sign_wrt_action: list[int], tau_value: int=1) -> int:
    """Returns the influence of memory to a choice of action.

    Computes the influence of memory to a choice of action as the sum of exponentially decaying stimulus from previous timesteps. The contributions of each stimulus to memory is dependent on the similarity of the current action to the action made at previous timesteps, with 1 for similar and -1 otherwise.

    Args:
        memory_tendency_stimulus: A list of previous stimulus.
        sign_wrt_action: A list of -1s and 1s indicating the relation of stimulus to the choice of action at the current timestep.
        tau_value: Steepness of the exponential decay of memory.
    """

    tendency_memory = 0.
    for time_idx, (tendency_idx, sign_idx) in (enumerate(zip(reversed(memory_tendency_stimulus), reversed(sign_wrt_action)))):
            tendency_memory += sign_idx * tendency_idx * np.exp(-(time_idx+1) / tau_value)

    return tendency_memory

def calculate_distance(target, sample, distance_type):
    """Calculates distance between target and sample, cutting the sample to the length of the target if necessary."""

    assert len(target) == len(sample), 'Target and sample size are not equal'

    if distance_type == "mse":
        return mean_squared_error(target, sample)
    elif distance_type == "mae":
        return mean_absolute_error(target, sample)
    else:
        raise ValueError("Type not supported")

def get_target(targe_type):
    if targe_type == 'guppy_single':
        df = pd.read_csv(f'{DATA_DIR}/target_data_single.csv', header=None)
        run_target = df[0].tolist()
    elif targe_type == 'guppy_serial':
        df = pd.read_csv(f'{DATA_DIR}/target_data_serial.csv', header=None)
        run_target = df[0].tolist()
    else:
        raise ValueError('Type not supported')

    return run_target

def initialize_experiment(config):
    experiment_type = config['EXPERIMENT_TYPE']
    simulation_type = config['SIMULATION_TYPE']

    # Get storage paths
    experiments_folder = 'experiments'
    if not os.path.isdir(experiments_folder):
        os.makedirs(experiments_folder)

    experiment_type_folder = f'{experiments_folder}/{experiment_type}'
    if not os.path.isdir(experiment_type_folder):
        os.makedirs(experiment_type_folder)

    simulation_type_folder = f'{experiments_folder}/{experiment_type}/{simulation_type}'
    if not os.path.isdir(simulation_type_folder):
        os.makedirs(simulation_type_folder)

    # Assign a unique run id based on records file and add it to config
    records_file = f'{simulation_type_folder}/records.csv'
    if not os.path.isfile(records_file):
        with open(records_file, 'w+') as recordsfile:
            recordsfile.write('RUN_ID,')
            for key in config.keys():
                recordsfile.write(key + ',')
            recordsfile.write('TIMESTAMP\n')

    df_records = pd.read_csv(records_file)
    if len(df_records) == 0:
        unique_runid = 1
    else:
        unique_runid = str(max(df_records['RUN_ID'])+1)
    config['RUN_ID'] = unique_runid

    # Add current time to config
    now = datetime.now()
    current_time = now.strftime('%Y%m%d_%H%M%S')
    config['TIMESTAMP'] = current_time

    # Make output folder
    output_folder = f'{simulation_type_folder}/output_{unique_runid}_{current_time}'

    return config, output_folder, records_file, df_records

def get_MAP(data):
    data = np.array(data)

    # Kernel Density Estimation
    kde = gaussian_kde(data.T)

    # Define a function to minimize (negative of the KDE)
    def neg_kde_density(x):
        return -kde(x)

    # Initial guess: mean of the samples
    initial_guess = np.mean(data, axis=0)

    # Optimization to find the mode (MAP estimate)
    result = minimize(neg_kde_density, initial_guess)#, method='Nelder-Mead')
    map_estimate = result.x

    return map_estimate