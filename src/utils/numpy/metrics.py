#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import multiprocessing as mp

from math import sqrt
from functools import partial
#from dask import delayed, compute
from src.utils.pytorch.losses import divide_no_nan

AVAILABLE_METRICS = ['mse', 'rmse', 'mape', 'smape', 'mase', 'rmsse',
                     'mini_owa', 'pinball_loss']

######################################################################
# METRICS
######################################################################

def mse(y, y_hat):
    """Calculates Mean Squared Error.

    MSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array
        predicted values

    Return
    ------
    scalar: MSE
    """
    mse = np.mean(np.square(y - y_hat))

    return mse

def rmse(y, y_hat):
    """Calculates Root Mean Squared Error.

    RMSE measures the prediction accuracy of a
    forecasting method by calculating the squared deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.
    Finally the RMSE will be in the same scale
    as the original time series so its comparison with other
    series is possible only if they share a common scale.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: RMSE
    """
    rmse = sqrt(np.mean(np.square(y - y_hat)))

    return rmse

def mape(y, y_hat):
    """Calculates Mean Absolute Percentage Error.

    MAPE measures the relative prediction accuracy of a
    forecasting method by calculating the percentual deviation
    of the prediction and the true value at a given time and
    averages these devations over the length of the series.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: MAPE
    """
    mape = np.mean(np.abs(y - y_hat) / np.abs(y))
    mape = 100 * mape

    return mape

def smape(y, y_hat):
    """Calculates Symmetric Mean Absolute Percentage Error.

    SMAPE measures the relative prediction accuracy of a
    forecasting method by calculating the relative deviation
    of the prediction and the true value scaled by the sum of the
    absolute values for the prediction and true value at a
    given time, then averages these devations over the length
    of the series. This allows the SMAPE to have bounds between
    0% and 200% which is desireble compared to normal MAPE that
    may be undetermined.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values

    Return
    ------
    scalar: SMAPE
    """
    delta_y = np.abs(y - y_hat)
    scale = np.abs(y) + np.abs(y_hat)
    smape = divide_no_nan(delta_y, scale)
    smape = 200 * np.mean(smape)
    assert smape <= 200, 'SMAPE should be lower than 200'

    return smape

def mae(y, y_hat, weights=None):
    """Calculates Mean Absolute Error.

    The mean absolute error 

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array
        predicted values
    weights: numpy array
        weights

    Return
    ------
    scalar: MAE
    """
    assert (weights is None) or (np.sum(weights)>0), 'Sum of weights cannot be 0'
    assert (weights is None) or (len(weights)==len(y)), 'Wrong weight dimension'
    mae = np.average(np.abs(y - y_hat), weights=weights)
    return mae

def mase(y, y_hat, y_train, seasonality=1):
    """Calculates the M4 Mean Absolute Scaled Error.

    MASE measures the relative prediction accuracy of a
    forecasting method by comparinng the mean absolute errors
    of the prediction and the true value against the mean
    absolute errors of the seasonal naive model.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values
    y_train: numpy array
      actual train values for Naive1 predictions
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1

    Return
    ------
    scalar: MASE
    """
    scale = np.mean(abs(y_train[seasonality:] - y_train[:-seasonality]))
    mase = np.mean(abs(y - y_hat)) / scale
    mase = 100 * mase

    return mase

def rmae(y, y_hat, y_train, seasonality=1):
    """Calculates Relative Mean Average Error.

    RMAE stands for the relative mean absolute error, it computes
    the mean absolute error and scales it relative to the mean absolute
    error of the Naive. 
    Similar to MASE, rMAE normalizes the MAE by the MAE of a naive forecast. 
    However, instead of considering the in-sample dataset, the naive forecast 
    is built based on the out-of-sample dataset.

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array
        predicted values
    y_train: numpy array
      actual train values for Naive1 predictions        

    Return
    ------
    scalar: RMAE
    """
    y_real = np.concatenate([y_train, y])

    y_hat_naive = y_real[-seasonality-len(y):-seasonality]

    scale = np.mean(abs(y - y_hat_naive))
    rmae = np.mean(abs(y - y_hat)) / scale

    return rmae

def rmsse(y, y_hat, y_train, seasonality=1):
    """Calculates the M5 Root Mean Squared Scaled Error.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1

    Return
    ------
    scalar: RMSSE
    """
    scale = np.mean(np.square(y_train[seasonality:] - y_train[:-seasonality]))
    rmsse = sqrt(mse(y, y_hat) / scale)
    rmsse = 100 * rmsse

    return rmsse

def mini_owa(y, y_hat, y_train, seasonality, y_bench):
    """Calculates the Overall Weighted Average for a single series.

    MASE, sMAPE for Naive2 and current model
    then calculatess Overall Weighted Average.

    Parameters
    ----------
    y: numpy array
        actual test values
    y_hat: numpy array of len h (forecasting horizon)
        predicted values
    seasonality: int
        main frequency of the time series
        Hourly 24,  Daily 7, Weekly 52,
        Monthly 12, Quarterly 4, Yearly 1
    y_train: numpy array
        insample values of the series for scale
    y_bench: numpy array of len h (forecasting horizon)
        predicted values of the benchmark model

    Return
    ------
    return: mini_OWA
    """
    mase_y = mase(y, y_hat, y_train, seasonality)
    mase_bench = mase(y, y_bench, y_train, seasonality)

    smape_y = smape(y, y_hat)
    smape_bench = smape(y, y_bench)

    mini_owa = ((mase_y/mase_bench) + (smape_y/smape_bench))/2

    return mini_owa

def pinball_loss(y: np.ndarray, y_hat: np.ndarray, tau: float=0.5, weights=None) -> np.ndarray:
    """Calculates the Pinball Loss.

    The Pinball loss measures the deviation of a quantile forecast.
    By weighting the absolute deviation in a non symmetric way, the
    loss pays more attention to under or over estimation.
    A common value for tau is 0.5 for the deviation from the median.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    tau: float
      Fixes the quantile against which the predictions are compared.
    Return
    ------
    return: pinball_loss
    """
    assert (weights is None) or (np.sum(weights)>0), 'Sum of weights cannot be 0'
    assert (weights is None) or (len(weights)==len(y)), 'Wrong weight dimension'

    delta_y = y - y_hat
    pinball = np.maximum(tau * delta_y, (tau - 1) * delta_y)
    pinball = np.average(pinball, weights=weights) #pinball.mean()
    return pinball

def panel_mape(y_hat):
    y_hat = y_hat.copy()
    y_hat['mape'] = np.abs(y_hat['y_hat']-y_hat['y'])/np.abs(y_hat['y'])
    y_hat_grouped = y_hat.groupby('unique_id').mean().reset_index()
    mape = np.mean(y_hat_grouped['mape'])
    return mape

def panel_smape(y_hat):
    y_hat = y_hat.copy()
    y_hat['smape'] = np.abs(y_hat['y_hat']-y_hat['y'])/(np.abs(y_hat['y'])+np.abs(y_hat['y_hat']))
    y_hat_grouped = y_hat.groupby('unique_id').mean().reset_index()
    smape = 2 * np.mean(y_hat_grouped['smape'])
    return smape