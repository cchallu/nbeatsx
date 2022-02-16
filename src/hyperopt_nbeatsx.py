import os
import argparse
import pickle
import glob
import itertools
import random
import torch
import numpy as np
import pandas as pd

from datetime import datetime
from functools import partial
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from src.utils.experiment.utils_experiment import run_val_nbeatsx, run_test_nbeatsx
from src.utils.data.datasets.epf import EPF, EPFInfo

TEST_DATE = {'NP': '2016-12-27',
             'PJM':'2016-12-27',
             'BE':'2015-01-04',
             'FR': '2015-01-04',
             'DE':'2016-01-04'}

def parse_trials(trials):
    """
    Parse trials object to a DataFrame.
    """
    # Initialize
    trials_dict = {'tid': [],
                   'loss': [],
                   'trajectories': [],
                   'mc': []}
    for tidx in range(len(trials)):
        # Main
        trials_dict['tid']  += [trials.trials[tidx]['tid']]
        trials_dict['loss'] += [trials.trials[tidx]['result']['loss']]
        trials_dict['trajectories'] += [trials.trials[tidx]['result']['trajectories']]

        # Model Configs
        mc = trials.trials[tidx]['result']['mc']
        trials_dict['mc'] += [mc]
    
    trials_df = pd.DataFrame(trials_dict)
    return trials_df

def get_experiment_space(args):
    """
    Defines the search space for hyperopt. The space depends on the type of model specified.
    For more information of each hyperparameter, refer to NBEATS model comments.
    """
    # Generic NBEATSx
    if args.space == 'nbeats_x':
        space = {'initialization':  hp.choice('initialization', ['orthogonal', 'he_normal', 'glorot_normal']),
                'activation': hp.choice('activation', ['softplus','selu','prelu','sigmoid']),
                'stack_types': hp.choice('stack_types', [ ['identity'],
                                                    1*['identity']+['exogenous_wavenet'],
                                                    ['exogenous_wavenet']+1*['identity'],
                                                    1*['identity']+['exogenous_tcn'],
                                                    ['exogenous_tcn']+1*['identity'] ]),
                'n_blocks': hp.choice('n_blocks', [ [1, 1] ]),
                'n_layers': hp.choice('n_layers', [ [2, 2] ]),
                'n_hidden_1': hp.quniform('n_hidden_1', 50, 500, 1),
                'n_hidden_2': hp.quniform('n_hidden_2', 50, 500, 1),
                'n_harmonics': hp.choice('n_harmonics', [0]),
                'n_polynomials': hp.choice('n_polynomials', [0]),
                'exogenous_n_channels': hp.quniform('exogenous_n_channels', 1, 10, 1),
                'batch_normalization': hp.choice('batch_normalization', [True, False]),
                'dropout_prob_theta': hp.uniform('dropout_prob_theta', 0, 1),
                'dropout_prob_exogenous': hp.uniform('dropout_prob_exogenous', 0, 0.5),
                'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.01)),
                'lr_decay': hp.choice('lr_decay', [0.5]),
                'n_lr_decay_steps': hp.choice('n_lr_decay_steps', [3]),
                'early_stopping': hp.choice('early_stopping', [10]),
                'eval_steps': hp.choice('eval_steps', [100]),
                'weight_decay': hp.choice('weight_decay', [0]),
                'n_iterations': hp.choice('n_iterations', [30_000]),
                'batch_size': hp.choice('batch_size', [256, 512]),
                'l1_theta': hp.choice('l1_theta', [0]), 
                'normalizer_y': hp.choice('normalizer_y', [None, 'median', 'invariant']),
                'normalizer_x': hp.choice('normalizer_x', [None, 'median', 'invariant']),
                'loss': hp.choice('loss', ['MAE']),
                'random_seed': hp.quniform('random_seed', 1, 1000, 1),
                'incl_pr1': hp.choice('incl_pr1', [True]),
                'incl_pr2': hp.choice('incl_pr2', [True, False]),
                'incl_pr3': hp.choice('incl_pr3', [True, False]),
                'incl_pr7': hp.choice('incl_pr7', [True, False]),
                'incl_ex1_0': hp.choice('incl_ex1_0', [True, False]),
                'incl_ex1_1': hp.choice('incl_ex1_1', [True, False]),
                'incl_ex1_7': hp.choice('incl_ex1_7', [True, False]),
                'incl_ex2_0': hp.choice('incl_ex2_0', [True, False]),
                'incl_ex2_1': hp.choice('incl_ex2_1', [True, False]),
                'incl_ex2_7': hp.choice('incl_ex2_7', [True, False]),
                'incl_day': hp.choice('incl_day', [True, False]),
                'n_val_weeks': hp.choice('n_val_weeks', [args.n_val_weeks])}
    if args.space == 'nbeats_x_interpretable':
        space = {'initialization':  hp.choice('initialization', ['orthogonal', 'he_normal', 'glorot_normal']),
                 'activation': hp.choice('activation', ['softplus','selu','prelu','sigmoid']),
                 'stack_types': hp.choice('stack_types', [ ['trend', 'seasonality', ],
                                                           ['trend', 'seasonality', 'exogenous_wavenet'],
                                                           ['exogenous_tcn', 'trend', 'seasonality'],
                                                           ['exogenous_wavenet', 'trend', 'seasonality'] ]),
                'n_blocks': hp.choice('n_blocks', [ [1, 1, 1] ]),
                'n_layers': hp.choice('n_layers', [ [2, 2, 2] ]),
                'n_hidden_1': hp.quniform('n_hidden_1', 50, 500, 1),
                'n_hidden_2': hp.quniform('n_hidden_2', 50, 500, 1),
                'n_harmonics': hp.choice('n_harmonics', [1, 2]),
                'n_polynomials': hp.choice('n_polynomials', [2, 3, 4]),
                'exogenous_n_channels': hp.quniform('exogenous_n_channels', 1, 10, 1),
                'batch_normalization': hp.choice('batch_normalization', [True, False]),
                'dropout_prob_theta': hp.uniform('dropout_prob_theta', 0, 1),
                'dropout_prob_exogenous': hp.uniform('dropout_prob_exogenous', 0, 0.5),
                'learning_rate': hp.loguniform('learning_rate', np.log(5e-4), np.log(0.1)),
                'lr_decay': hp.choice('lr_decay', [0.5]),
                'n_lr_decay_steps': hp.choice('n_lr_decay_steps', [3]),
                'early_stopping': hp.choice('early_stopping', [10]),
                'eval_steps': hp.choice('eval_steps', [100]),
                'weight_decay': hp.choice('weight_decay', [0]),
                'n_iterations': hp.choice('n_iterations', [30_000]),
                'batch_size': hp.choice('batch_size', [256, 512]),
                'l1_theta': hp.choice('l1_theta', [0]), 
                'normalizer_y': hp.choice('normalizer_y', [None, 'median', 'invariant']),
                'normalizer_x': hp.choice('normalizer_x', [None, 'median', 'invariant']),
                'loss': hp.choice('loss', ['MAE']),
                'random_seed': hp.quniform('random_seed', 1, 1000, 1),
                'incl_pr1': hp.choice('incl_pr1', [True]),
                'incl_pr2': hp.choice('incl_pr2', [True, False]),
                'incl_pr3': hp.choice('incl_pr3', [True, False]),
                'incl_pr7': hp.choice('incl_pr7', [True, False]),
                'incl_ex1_0': hp.choice('incl_ex1_0', [True, False]),
                'incl_ex1_1': hp.choice('incl_ex1_1', [True, False]),
                'incl_ex1_7': hp.choice('incl_ex1_7', [True, False]),
                'incl_ex2_0': hp.choice('incl_ex2_0', [True, False]),
                'incl_ex2_1': hp.choice('incl_ex2_1', [True, False]),
                'incl_ex2_7': hp.choice('incl_ex2_7', [True, False]),
                'incl_day': hp.choice('incl_day', [True, False]),
                'n_val_weeks': hp.choice('n_val_weeks', [args.n_val_weeks])}
    return space

def main(args):
    # Random seeds
    np.random.seed(1)
    random.seed(1)

    #---------------------------------------------- Directories ----------------------------------------------#
    output_dir = f'./results/{args.dataset}/{args.space}/'
    os.makedirs(output_dir, exist_ok = True)
    assert os.path.exists(output_dir), f'Output dir {output_dir} does not exist'

    if args.experiment_id is None:
        experiment_id = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    else:
        experiment_id = args.experiment_id

    hyperopt_file = output_dir + f'hyperopt_{experiment_id}.p'
    result_test_file = output_dir + f'result_test_{experiment_id}.p'

    #---------------------------------------------- Read  Data ----------------------------------------------#
    print('\n'+75*'-')
    print(28*'-', 'Preparing Dataset', 28*'-')
    print(75*'-'+'\n')

    test_date = TEST_DATE[args.dataset]
    Y_df, X_df, _ = EPF.load_groups(directory='./data', groups=[args.dataset])
    
    # Remove test set
    test_date = TEST_DATE[args.dataset]
    y_insample_df = Y_df[Y_df['ds']<test_date].reset_index(drop=True)
    X_t_insample_df = X_df[X_df['ds']<test_date].reset_index(drop=True)
    y_outsample_df = Y_df[Y_df['ds']>=test_date].reset_index(drop=True)
    X_t_outsample_df = X_df[X_df['ds']>=test_date].reset_index(drop=True)
    

    print(f'Dataset: {args.dataset}')
    print('X: time series features, of shape (#hours, #times,#features): \t' + str(X_t_insample_df.shape))
    print('Y: target series (in X), of shape (#hours, #times): \t \t' + str(y_insample_df.shape))
    print('\n')

    #-------------------------------------- Hyperparameter Optimization --------------------------------------#
    if not os.path.isfile(hyperopt_file):
        print('\n'+75*'-')
        print(22*'-', 'Start Hyperparameter  tunning', 22*'-')
        print(75*'-'+'\n')

        space = get_experiment_space(args)

        trials = Trials()
        fmin_objective = partial(run_val_nbeatsx, Y_df=y_insample_df, X_df=X_t_insample_df,
                                 data_augmentation=args.data_augmentation,
                                 random_validation=args.random_validation,
                                 trials=trials, trials_file_name=hyperopt_file)
        fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=args.hyperopt_iters, trials=trials, verbose=True)

        # Save output
        with open(hyperopt_file, "wb") as f:
            pickle.dump(trials, f)
    else:
        print('Hyperparameter tunning already performed')

    print('\n'+75*'-')
    print(20*'-', 'Hyperparameter  tunning  finished', 20*'-')
    print(75*'-'+'\n')

    #-------------------------------------- Best model in test --------------------------------------#
    # Read and parse trials pickle
    trials = pickle.load(open(hyperopt_file, 'rb'))
    trials_df = parse_trials(trials)

    # Get best mc
    idx = trials_df.loss.idxmin()
    best_mc = trials_df.loc[idx]['mc']
    print("Best loss: ", trials_df.loc[idx]['loss'])

    # Append train and test data
    Y_df = y_insample_df.append(y_outsample_df, ignore_index=True)
    Y_df = Y_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    X_df = X_t_insample_df.append(X_t_outsample_df, ignore_index=True)
    X_df = X_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    # Run auxiliary function which performs rolling forecasts
    result_test = run_test_nbeatsx(mc=best_mc, Y_df=Y_df, X_df=X_df, len_outsample=len(y_outsample_df))

    # Save output
    with open(result_test_file, "wb") as f:
        pickle.dump(result_test, f)

def parse_args():
    desc = "NBEATSx run experiment"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, required=True, help='Name of market')
    parser.add_argument('--space', type=str, required=True, help='Name of model')
    parser.add_argument('--data_augmentation', type=int, required=True, help='Data augmentation flag')
    parser.add_argument('--random_validation', type=int, required=True, help='Random validation flag')
    parser.add_argument('--n_val_weeks', type=int, required=True, help='Val weeks')
    parser.add_argument('--hyperopt_iters', type=int, help='hyperopt_iters')
    parser.add_argument('--experiment_id', default=None, required=False, type=str, help='string to identify experiment')
    return parser.parse_args()

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    main(args)

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate nbeatsx
# PYTHONPATH=. python src/hyperopt_nbeatsx.py --dataset 'NP' --space "nbeats_x" --data_augmentation 0 --random_validation 0 --n_val_weeks 52 --hyperopt_iters 2 --experiment_id "20210129_0_0"
