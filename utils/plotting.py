from src.utils.math import step_function 
from typing import List
import numpy as np
import polars as pl
from scipy import stats

#TODO: Read this from config later
#TODO: Read the data dynamically using read methods in io
MAX_RANGE = 300000

def read_data(exp_ids: List):
    all_data = []
    for _, id in enumerate(exp_ids):
        df = pl.read_csv(f'index/results/data/{id}.csv')

        x,y = df['num_evaluations'].to_numpy(), df['best_reward'].to_numpy()
        to_plot = step_function(x,y, MAX_RANGE)
        all_data.append(to_plot)

    return all_data


def extract_exps_per_flag(manager, flag):
    flag_dict = {}
    for exp in manager.get_experiment_desc():
        #exp_ids = manager.get_experiment_ids(exp)
        
        dict_exp = dict(exp)
        flag_val = dict_exp[flag]

        if flag_val not in flag_dict:
            flag_dict[flag_val] = []

        flag_dict[flag_val].append(exp)

    return flag_dict
      

def single_id_to_plot(exp_ids: List, plotting_script: callable):
    for _, id in enumerate(exp_ids):
        df = pl.read_csv(f'index/results/data/{id}.csv')
        x,y = df['num_evaluations'].to_numpy(), df['best_reward'].to_numpy()
        to_plot = step_function(x,y, MAX_RANGE)

        plotting_script(to_plot)

def mean_and_confidence_interval(exp_ids: List, plotting_script: callable):
    
    seeds = []
    for _, id in enumerate(exp_ids):
        df = pl.read_csv(f'index/results/data/{id}.csv')

        x,y = df['num_evaluations'].to_numpy(), df['best_reward'].to_numpy()
        seed = step_function(x,y, MAX_RANGE)

        seeds.append(seed)

    data = np.stack(seeds, axis=0).astype(np.float64, copy=False) ##Turn to numpy array.
    res = stats.bootstrap(
    (data,),
    np.mean,                 # statistic
    axis=0,                  # resample along the samples axis
    n_resamples=100,        # adjust to your budget
    confidence_level=0.95,
    vectorized=True,         # np.mean supports axis
    method="percentile",            # better coverage than plain percentile, esp. with n=20
    random_state=42
            )

    to_plot = np.mean(seeds, axis = 0)
    ci_lower = res.confidence_interval.low
    ci_upper = res.confidence_interval.high
        
    plotting_script(to_plot, ci_lower, ci_upper)
     

def aggregate_id_to_plot(exp_ids: List, plotting_script: callable):
    
    list_of_ids = []

    for _, id in enumerate(exp_ids):
        df = pl.read_csv(f'index/results/data/{id}.csv')
        x,y = df['num_evaluations'].to_numpy(), df['best_reward'].to_numpy()

        to_plot = step_function(x,y, MAX_RANGE)
        list_of_ids.append(to_plot)

    list_of_ids = np.array(list_of_ids)
    to_plot = np.mean(list_of_ids, axis = 0)
    plotting_script(to_plot)


def experiment_title(exp: tuple):
      return '\n'.join(f"{k}: {v}" for k, v in exp)