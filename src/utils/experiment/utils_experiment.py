import os
import time
import pickle
import glob
import argparse
import itertools
import random

import numpy as np
import pandas as pd
import torch as t

from datetime import datetime
from hyperopt import STATUS_OK

from src.utils.numpy.metrics import rmae, mae, mape, smape, rmse
from src.utils.data.utils import Scaler
from src.utils.pytorch.ts_dataset import TimeSeriesDataset
from src.utils.pytorch.ts_loader import TimeSeriesLoader
from src.nbeats.nbeats import Nbeats


def transform_data(Y_df, X_df, mask, normalizer_y, normalizer_x):
    """
    Scales Y_df and X_df with normalizers for y and x. For computing scales, only observations defined
    as 1 in the mask vector will be used.
    """
    y_shift = None
    y_scale = None

    mask = mask.astype(int)
    
    if normalizer_y is not None:
        scaler_y = Scaler(normalizer=normalizer_y)
        Y_df['y'] = scaler_y.scale(x=Y_df['y'].values, mask=mask)
    else:
        scaler_y = None

    if normalizer_x is not None:
        scaler_x = Scaler(normalizer=normalizer_x)
        X_df['Exogenous1'] = scaler_x.scale(x=X_df['Exogenous1'].values, mask=mask)

        scaler_x = Scaler(normalizer=normalizer_x)
        X_df['Exogenous2'] = scaler_x.scale(x=X_df['Exogenous2'].values, mask=mask)

    filter_variables = ['unique_id', 'ds', 'Exogenous1', 'Exogenous2', 'week_day'] + [col for col in X_df if (col.startswith('day'))]

    X_df = X_df[filter_variables]

    return Y_df, X_df, scaler_y

def train_val_split(len_series, offset, window_sampling_limit, n_val_weeks, ds_per_day):
    """
    Returns train and validation indices (of the time series). Randomly selects n_val_weeks
    as validation.
    """
    last_ds = len_series - offset
    first_ds = max(last_ds - window_sampling_limit, 0)

    last_day = int(last_ds/ds_per_day)
    first_day = int(first_ds/ds_per_day)

    days = set(range(first_day, last_day)) # All days, to later get train days
    # Sample weeks from here, -7 to avoid sampling from last week
    # To not sample first week and have inputs
    sampling_days = set(range(first_day + 7, last_day - 7))
    validation_days = set({}) # Val days set
    
    # For loop for n of weeks in validation
    for i in range(n_val_weeks):
        # Sample random day, init of week
        init_day = random.sample(sampling_days, 1)[0]
        # Select days of sampled init of week
        sampled_days = list(range(init_day, min(init_day+7, last_day)))
        # Add days to validation days
        validation_days.update(sampled_days)
        # Remove days from sampling_days, including overlapping resulting previous week
        days_to_remove = set(range(init_day-6, min(init_day+7, last_day)))
        sampling_days = sampling_days.difference(days_to_remove)

    train_days = days.difference(validation_days)

    train_days = sorted(list(train_days))
    validation_days = sorted(list(validation_days))

    train_idx = []
    for day in train_days:
        hours_idx = range(day*ds_per_day,(day+1)*ds_per_day)
        train_idx += hours_idx

    val_idx = []
    for day in validation_days:
        hours_idx = range(day*ds_per_day,(day+1)*ds_per_day)
        val_idx += hours_idx

    assert all([idx < last_ds for idx in val_idx]), 'Last idx should be smaller than last_ds'
    
    return train_idx, val_idx

def run_val_nbeatsx(hyperparameters, Y_df, X_df, data_augmentation, random_validation, trials, trials_file_name):
    """
    Auxiliary function to run NBEATSx for hyperopt hyperparameter optimization.
    Return a dictionary with loss and relevant information.
    """

    # To not modify Y_df and X_df
    Y_df_scaled = Y_df.copy()
    X_df_scaled = X_df.copy()

    # Save trials, can analyze progress
    save_every_n_step = 5
    current_step = len(trials.trials)
    if (current_step % save_every_n_step==0):
        with open(trials_file_name, "wb") as f:
            pickle.dump(trials, f)

    start_time = time.time()
    
    # -------------------------------------------------- Parse hyperparameters --------------------------------------------------
    # Models and loaders will receive hyperparameters from mc (model config) dictionary.
    mc = hyperparameters

    if data_augmentation:
        mc['idx_to_sample_freq'] = 1
    else:
        mc['idx_to_sample_freq'] = 24

    # Avoid this combination because it can produce results with large variance
    if (mc['batch_normalization']) and (mc['normalizer_y']==None):
         mc['normalizer_y'] = 'median'

    # Other hyperparameters which we do not explore (are fixed)
    mc['input_size_multiplier'] = 7
    mc['output_size'] = 24
    mc['window_sampling_limit_multiplier'] = 365*4
    mc['shared_weights'] = False
    mc['x_s_n_hidden'] = 0
    mc['train_every_n_steps'] = 1
    mc['seasonality'] = 24
    mc['loss_hypar'] = None
    mc['val_loss'] = mc['loss']

    mc['n_hidden'] = len(mc['stack_types']) * [ [int(mc['n_hidden_1']), int(mc['n_hidden_2'])] ]

    # This dictionary will be used to select particular lags as inputs for each y and exogenous variables.
    # For eg, -1 will include the future (corresponding to the forecasts variables), -2 will add the last
    # available day (1 day lag), etc.
    include_var_dict = {'y': [],
                        'Exogenous1': [],
                        'Exogenous2': [],
                        'week_day': []}

    if mc['incl_pr1']: include_var_dict['y'].append(-2)
    if mc['incl_pr2']: include_var_dict['y'].append(-3)
    if mc['incl_pr3']: include_var_dict['y'].append(-4)
    if mc['incl_pr7']: include_var_dict['y'].append(-8)
        
    if mc['incl_ex1_0']: include_var_dict['Exogenous1'].append(-1)
    if mc['incl_ex1_1']: include_var_dict['Exogenous1'].append(-2)
    if mc['incl_ex1_7']: include_var_dict['Exogenous1'].append(-8)
        
    if mc['incl_ex2_0']: include_var_dict['Exogenous2'].append(-1)
    if mc['incl_ex2_1']: include_var_dict['Exogenous2'].append(-2)
    if mc['incl_ex2_7']: include_var_dict['Exogenous2'].append(-8)

    # Inside the model only the week_day of the first hour of the horizon will be selected as input
    if mc['incl_day']: include_var_dict['week_day'].append(-1) 

    print(47*'=' + '\n')
    print(pd.Series(mc))
    print(47*'=' + '\n')
    
    # -------------------------------------------------- Train and Validation Mask --------------------------------------------------
    # train_mask: 1 to keep, 0 to hide from training
    train_outsample_mask = np.ones(len(Y_df), dtype=int)
    if random_validation:
        print('Random validation activated')
        # Set seed again to have same validation windows on each run
        np.random.seed(1)
        random.seed(1)
        _, val_idx = train_val_split(len_series=len(Y_df), offset=0,
                                window_sampling_limit= mc['window_sampling_limit_multiplier'] * mc['output_size'],
                                n_val_weeks = mc['n_val_weeks'], ds_per_day=24)
        train_outsample_mask[val_idx] = 0
    else:
        print('Random validation de-activated')
        # Last mc['n_val_weeks'] * 7 days will be used as validation
        train_outsample_mask[-mc['n_val_weeks'] * 7 * mc['output_size']:] = 0

    print(f'Train {sum(train_outsample_mask)} hours = {np.round(sum(train_outsample_mask)/(24*365),2)} years')
    print(f'Validation {sum(1-train_outsample_mask)} hours = {np.round(sum(1-train_outsample_mask)/(24*365),2)} years')

    # To compute validation loss in true scale
    y_validation_vector = Y_df['y'].values[(1-train_outsample_mask)==1]

    # -------------------------------------------------- Data Wrangling --------------------------------------------------
    # Transform data with scale transformation
    Y_df_scaled, X_df_scaled, scaler_y = transform_data(Y_df = Y_df_scaled,
                                                        X_df = X_df_scaled,
                                                        mask = train_outsample_mask,
                                                        normalizer_y = mc['normalizer_y'],
                                                        normalizer_x = mc['normalizer_x'])

    # Dataset object. Pre-process the DataFrame into pytorch tensors and windows.
    ts_dataset = TimeSeriesDataset(Y_df=Y_df_scaled, X_df=X_df_scaled, ts_train_mask=train_outsample_mask)

    # Loaders object. Sample windows of dataset object.
    # For more information on each parameter, refer to comments on Loader object.
    train_ts_loader = TimeSeriesLoader(model='nbeats',
                                       ts_dataset=ts_dataset,
                                       window_sampling_limit=mc['window_sampling_limit_multiplier'] * mc['output_size'],
                                       offset=0,
                                       input_size=int(mc['input_size_multiplier'] * mc['output_size']),
                                       output_size=int(mc['output_size']),
                                       idx_to_sample_freq=int(mc['idx_to_sample_freq']),
                                       batch_size=int(mc['batch_size']),
                                       is_train_loader=True,
                                       shuffle=True)

    # Will sample windows on the validation set for early stopping.
    val_ts_loader = TimeSeriesLoader(model='nbeats',
                                     ts_dataset=ts_dataset,
                                     window_sampling_limit=mc['window_sampling_limit_multiplier'] * mc['output_size'],
                                     offset=0,
                                     input_size=int(mc['input_size_multiplier'] * mc['output_size']),
                                     output_size=int(mc['output_size']),
                                     idx_to_sample_freq=24, #TODO: pensar esto
                                     batch_size=int(mc['batch_size']),
                                     is_train_loader=False,
                                     shuffle=False)

    mc['include_var_dict'] = include_var_dict
    mc['t_cols'] = ts_dataset.t_cols

    # -------------------------------------------------- Instantiate model,fit and predict --------------------------------------------------
    # Instantiate and train model
    model = Nbeats(input_size_multiplier=mc['input_size_multiplier'],
                   output_size=int(mc['output_size']),
                   shared_weights=mc['shared_weights'],
                   initialization=mc['initialization'],
                   activation=mc['activation'],
                   stack_types=mc['stack_types'],
                   n_blocks=mc['n_blocks'],
                   n_layers=mc['n_layers'],
                   n_hidden=mc['n_hidden'],
                   n_harmonics=int(mc['n_harmonics']),
                   n_polynomials=int(mc['n_polynomials']),
                   x_s_n_hidden = int(mc['x_s_n_hidden']),
                   exogenous_n_channels=int(mc['exogenous_n_channels']),
                   include_var_dict=mc['include_var_dict'],
                   t_cols=mc['t_cols'],
                   batch_normalization = mc['batch_normalization'],
                   dropout_prob_theta=mc['dropout_prob_theta'],
                   dropout_prob_exogenous=mc['dropout_prob_exogenous'],
                   learning_rate=float(mc['learning_rate']),
                   lr_decay=float(mc['lr_decay']),
                   n_lr_decay_steps=float(mc['n_lr_decay_steps']),
                   early_stopping=int(mc['early_stopping']),
                   weight_decay=mc['weight_decay'],
                   l1_theta=mc['l1_theta'],
                   n_iterations=int(mc['n_iterations']),
                   loss=mc['loss'],
                   loss_hypar=mc['loss_hypar'],
                   val_loss=mc['val_loss'],
                   seasonality=int(mc['seasonality']),
                   random_seed=int(mc['random_seed']))

    # Fit model
    model.fit(train_ts_loader=train_ts_loader, val_ts_loader=val_ts_loader, n_iterations=mc['n_iterations'], eval_steps=mc['eval_steps'])
    
    # Predict on validation
    _, y_hat, _ = model.predict(ts_loader=val_ts_loader)
    y_hat = y_hat.flatten()

    # Scale to original scale
    if mc['normalizer_y'] is not None:
        y_hat = scaler_y.inv_scale(x=y_hat)

    # Compute MAE
    val_mae = mae(y=y_validation_vector, y_hat=y_hat)
    run_time = time.time() - start_time

    results =  {'loss': val_mae,
                'mc': mc,
                'final_insample_loss': model.final_insample_loss,
                'final_outsample_loss': model.final_outsample_loss,
                'trajectories': model.trajectories,
                'run_time': run_time,
                'status': STATUS_OK}

    return results

def run_test_nbeatsx(mc, Y_df, X_df, len_outsample):
    """
    Auxiliary function to produce rolling forecast and re-calibration of the NBEATSx model on the test set.
    """

    print(47*'=' + '\n')
    print(pd.Series(mc))
    print(47*'=' + '\n')

    # -------------------------------------------------- Rolling prediction on test --------------------------------------------------
    # Each split is 1 day
    n_splits = int(len_outsample/mc['output_size'])
    print(f'Number of splits: {n_splits}')
    
    start_time = time.time()
    y_hat = []
    y_hat_decomposed = []
    split_info = []
    for split in range(n_splits):
        print(10*'-', f'Split {split+1}/{n_splits}', 10*'-')
        # The offset can be interpreted as the timestamps in test (all hours of the remaining days) Eg. if split=0 (first day of test),
        # offset will be equal to 728*24. The offset is then used to filter the last part of the data, so that the model is trained
        # with the information prior to the day currently being predicted.
        offset = len_outsample - split * mc['output_size']
        assert offset > 0, 'Offset must be positive'
        print(f'Offset: {offset}')

        if (split % mc['train_every_n_steps'] > 0):
            recalibrate_model = False
            print('Model not recalibrated')
        else:
            recalibrate_model = True
            print(f'Model recalibrated')

        # -------------------------------------------------- Data wrangling --------------------------------------------------
        Y_df_scaled = Y_df.copy()
        X_df_scaled = X_df.copy()

        # train_mask: 1 to keep, 0 to mask
        scaler_mask = np.ones(len(Y_df_scaled))
        scaler_mask[-offset:] = 0
        Y_df_scaled, X_df_scaled, scaler_y = transform_data(Y_df=Y_df_scaled, X_df=X_df_scaled, mask=scaler_mask,
                                                            normalizer_y=mc['normalizer_y'], normalizer_x=mc['normalizer_x'])
        
        # Train-val split for early stopping. Validation set are n_val_weeks selected at random.
        _, val_idx = train_val_split(len_series=len(Y_df), offset=offset,
                                     window_sampling_limit= mc['window_sampling_limit_multiplier'] * mc['output_size'],
                                     n_val_weeks = mc['n_val_weeks'], ds_per_day=24)

        # train_mask: 1 to keep, 0 to mask
        train_outsample_mask = np.ones(len(Y_df_scaled))
        train_outsample_mask[val_idx] = 0

        # Instantiate train and validation dataset and loaders
        ts_dataset = TimeSeriesDataset(Y_df=Y_df_scaled, X_df=X_df_scaled, ts_train_mask=train_outsample_mask)
   
        train_ts_loader = TimeSeriesLoader(model='nbeats',
                                           ts_dataset=ts_dataset,
                                           window_sampling_limit=mc['window_sampling_limit_multiplier'] * mc['output_size'],
                                           offset=offset, # TO FILTER LAST OFFSET TIME STAMPS
                                           input_size=int(mc['input_size_multiplier'] * mc['output_size']),
                                           output_size=int(mc['output_size']),
                                           idx_to_sample_freq=int(mc['idx_to_sample_freq']),
                                           batch_size=int(mc['batch_size']),
                                           is_train_loader=True,
                                           shuffle=True)

        val_ts_loader = TimeSeriesLoader(model='nbeats',
                                         ts_dataset=ts_dataset,
                                         window_sampling_limit=mc['window_sampling_limit_multiplier'] * mc['output_size'],
                                         offset=offset, # TO FILTER LAST OFFSET TIME STAMPS
                                         input_size=int(mc['input_size_multiplier'] * mc['output_size']),
                                         output_size=int(mc['output_size']),
                                         idx_to_sample_freq=24,
                                         batch_size=int(mc['batch_size']),
                                         is_train_loader=False,
                                         shuffle=False)

        # Test dataset and loader, to sample window with the day currently being predicted
        # Test mask: 1s for 24 lead time
        test_mask = np.zeros(len(Y_df_scaled))
        test_mask[-offset:] = 1
        test_mask[(len(Y_df_scaled) - offset + mc['output_size']):] = 0

        assert test_mask.sum() == mc['output_size'], f'Sum of Test mask must be {mc["output_size"]} not {test_mask.sum()}'

        ts_dataset_test = TimeSeriesDataset(Y_df=Y_df_scaled, X_df=X_df_scaled, ts_train_mask=test_mask)
        test_ts_loader = TimeSeriesLoader(model='nbeats',
                                          ts_dataset=ts_dataset_test,
                                          window_sampling_limit=mc['window_sampling_limit_multiplier'] * mc['output_size'],
                                          offset=offset - mc['output_size'], # To bypass leakeage protection
                                          input_size=int(mc['input_size_multiplier'] * mc['output_size']),
                                          output_size=int(mc['output_size']),
                                          idx_to_sample_freq=24,
                                          batch_size=int(mc['batch_size']),
                                          is_train_loader=True,
                                          shuffle=False)

        # ------------------------------------ Instantiate model,fit and predict ------------------------------------
        # Re-initialize model if recalibration is needed in this step
        if recalibrate_model:
            # Instantiate and train model
            model = Nbeats(input_size_multiplier=mc['input_size_multiplier'],
                           output_size=int(mc['output_size']),
                           shared_weights=mc['shared_weights'],
                           initialization=mc['initialization'],
                           activation=mc['activation'],
                           stack_types=mc['stack_types'],
                           n_blocks=mc['n_blocks'],
                           n_layers=mc['n_layers'],
                           n_hidden=mc['n_hidden'],
                           n_harmonics=int(mc['n_harmonics']),
                           n_polynomials=int(mc['n_polynomials']),
                           x_s_n_hidden = int(mc['x_s_n_hidden']),
                           exogenous_n_channels=int(mc['exogenous_n_channels']),
                           include_var_dict=mc['include_var_dict'],
                           t_cols=mc['t_cols'],
                           batch_normalization = mc['batch_normalization'],
                           dropout_prob_theta=mc['dropout_prob_theta'],
                           dropout_prob_exogenous=mc['dropout_prob_exogenous'],
                           learning_rate=float(mc['learning_rate']),
                           lr_decay=float(mc['lr_decay']),
                           n_lr_decay_steps=float(mc['n_lr_decay_steps']),
                           early_stopping=int(mc['early_stopping']),
                           weight_decay=mc['weight_decay'],
                           l1_theta=mc['l1_theta'],
                           n_iterations=int(mc['n_iterations']),
                           loss=mc['loss'],
                           loss_hypar=mc['loss_hypar'],
                           val_loss=mc['val_loss'],
                           seasonality=int(mc['seasonality']),
                           random_seed=int(mc['random_seed']))

            model.fit(train_ts_loader=train_ts_loader, val_ts_loader=val_ts_loader,
                      n_iterations=mc['n_iterations'], eval_steps=mc['eval_steps'])

        # Predict with re-calibrated model in test day
        _, y_hat_split, y_hat_decomposed_split, _ = model.predict(ts_loader=test_ts_loader,
                                                                  return_decomposition=True)
        y_hat_split = y_hat_split.flatten() # Only for univariate models

        assert len(y_hat_split) == mc['output_size'], 'Forecast should have length equal to output_size'
 
        if mc['normalizer_y'] is not None:
            y_hat_split = scaler_y.inv_scale(x=y_hat_split)

        print('Prediction: ', y_hat_split)
        y_hat += list(y_hat_split)
        y_hat_decomposed.append(y_hat_decomposed_split)
        split_info.append(model.trajectories)
        print('y_hat_decomposed', y_hat_decomposed)

    run_time = time.time() - start_time
    print(10*'-', f'Time: {run_time} s', 10*'-')

    # Output evaluation
    evaluation_dict = {'y_hat': y_hat,
                       'y_hat_decomposed': y_hat_decomposed,
                       'split_info': split_info,
                       'run_time': run_time}

    return evaluation_dict