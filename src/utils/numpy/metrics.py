import numpy as np
import pandas as pd
import multiprocessing as mp

from math import sqrt
from functools import partial

from src.utils.pytorch.losses import divide_no_nan
from scipy import stats
from scipy.stats.distributions import chi2

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['font.family'] = 'serif'

AVAILABLE_METRICS = ['mse', 'rmse', 'mape', 'smape', 'mase', 'rmsse', 'gwtest',
                     'mini_owa', 'pinball_loss']
FONTSIZE = 20

######################################################################
# SIGNIFICANCE TEST
######################################################################

def Newey_West(Z, n_lags):
    """ Newey-West HAC estimator
    Parameters
    ----------
    Z: (n, k) ndarray
    n_lags: int
        number of lags to consider as available information.

    Returns
    -------
    omega_hat: Newey-West HAC estimator of the covariance matrix
    """

    assert n_lags > 0

    n, k = Z.shape

    Z = Z - np.ones((n, 1)) * np.mean(Z, axis=0)
    gamma = -999 * np.ones((n_lags, k))
    omega_hat = (1/n) * np.matmul(np.transpose(Z), Z)

    Zlag = np.array([np.pad(Z, ((i,0), (0,0)), mode='constant', 
                            constant_values = 0)[:n] 
                     for i in range(1, n_lags + 1)])
    gamma = (1/n) * (np.matmul(np.transpose(Z), Zlag) + 
            np.matmul(np.einsum('ijk -> ikj', Zlag), Z))
    weights = 1 - np.array(range(1,n_lags + 1))/(n_lags + 1)
    omega_hat = omega_hat + \
                np.sum(gamma * np.expand_dims(weights, 
                                              axis = (1,2)), 
                                              axis = 0)
    return omega_hat

def GW_CPA_test(loss1: np.ndarray, 
                loss2: np.ndarray, 
                tau: int,
                alpha: float=0.05,
                conditional: bool=False,
                verbose: bool=True):
    """ 
    Giacomini-White Conditional Predictive Ability Test
    Parameters
    ----------
    loss1: numpy array
        losses of model 1
    loss2: numpy array
        losses of model 2
    tau: int
        the past information treated as 'available' for the test.
    unconditional: boolean, 
        True if unconditional (DM test), False if conditional (GW test).
    verbose: boolean, 
        True if prints of test are needed

    Returns
    -------
    test_stat: test statistic of the conditional predictive ability test
    crit_val: critical value of the chi-square test for a 5% confidence level
    p-vals: (k,) p-value of the test
    """   

    assert len(loss1) == len(loss2)

    lossdiff = loss1 - loss2
    t = len(loss1)
    instruments = np.ones_like(loss1)

    if conditional:
        instruments = np.hstack((instruments[:t-tau], 
                                 lossdiff[:-tau]))
        lossdiff = lossdiff[tau:]
        t = t - tau

    reg = instruments * lossdiff
    
    if tau == 1:
        res_beta = np.linalg.lstsq(reg, np.ones((t)), rcond=None)[0]

        err = np.ones((t,1)) - reg.dot(res_beta)
        r2 = 1 - np.mean(err**2)
        test_stat = t * r2
    
    else:

        zbar = np.mean(reg, axis=0)
        n_lags = tau - 1
        omega = Newey_West(Z=reg, n_lags=n_lags)
        test_stat = np.expand_dims(t*zbar, 
                                   axis=0).dot(np.linalg.inv(omega)).\
                                   dot(zbar)
      
    test_stat *= np.sign(np.mean(lossdiff))
    
    q = reg.shape[1]
    crit_val = chi2.ppf(1-alpha, df=q)
    p_val = 1 - chi2.cdf(test_stat, q)

    av_diff_loss = np.mean(loss1-loss2)
    s = '+' if np.mean(loss1-loss2) > 0 else '-'
    
    if verbose:
        if conditional: print('\nConditional test:\n')
        if not conditional: print('\nUnconditional test:\n')
        print(f'Forecast horizon: {tau}, Nominal Risk Level: {alpha}')
        print(f'Test-statistic: {test_stat} ({s})')
        print(f'Critical value: {crit_val}')
        print(f'p-value: {p_val}\n')
    
    return test_stat, crit_val, p_val



def gwtest(loss1, loss2, tau=1, conditional=1):
    d = loss1 - loss2
    TT = np.max(d.shape)

    if conditional:
        instruments = np.stack([np.ones_like(d[:-tau]), d[:-tau]])
        d = d[tau:]
        T = TT - tau
    else:
        instruments = np.ones_like(d)
        T = TT
    
    instruments = np.array(instruments, ndmin=2)

    reg = np.ones_like(instruments) * -999
    for jj in range(instruments.shape[0]):
        reg[jj, :] = instruments[jj, :] * d
    
    if tau == 1:
        # print(reg.shape, T)
        # print(reg.T)        
        betas = np.linalg.lstsq(reg.T, np.ones(T), rcond=None)[0]
        err = np.ones((T, 1)) - np.dot(reg.T, betas)
        r2 = 1 - np.mean(err**2)
        GWstat = T * r2
    else:
        raise NotImplementedError
        zbar = np.mean(reg, -1)
        nlags = tau - 1
        # ...
    
    GWstat *= np.sign(np.mean(d))
    # pval = 1 - scipy.stats.norm.cdf(GWstat)
    # if np.isnan(pval) or pval > .1:
    #     pval = .1
    # return pval
    
    q = reg.shape[0]
    pval = 1 - stats.chi2.cdf(GWstat, q)
    # if np.isnan(pval) or pval > .1:
    #     pval = .1
    return pval

def get_nbeatsx_cmap():
    cmap = cm.get_cmap('pink', 512)
    yellows = cmap(np.linspace(0.5, 0.95, 256))

    cmap = cm.get_cmap('Blues', 256)
    blues = cmap(np.linspace(0.45, 0.75, 256))

    newcolors = np.concatenate([yellows, blues])

    #extra = np.array([116/256, 142/256, 157/256, 1])
    extra = np.array([66/256, 75/256, 98/256, 1])
    #extra = np.array([3/256, 34/256, 71/256, 1])
    newcolors[-10:, :] = extra
    newcmap = ListedColormap(newcolors)
    return newcmap

def get_epftoolbox_cmap():
    cmap = cm.get_cmap('YlGn_r', 512)
    #yellows = cmap(np.linspace(0.65, 1.0, 256))
    yellows = cmap(np.linspace(0.6, 1.0, 256))

    cmap = cm.get_cmap('gist_heat_r', 256)
    #reds = cmap(np.linspace(0.55, 0.75, 256))
    reds = cmap(np.linspace(0.39, 0.66, 256))

    newcolors = np.concatenate([yellows, reds])

    #extra = np.array([116/256, 142/256, 157/256, 1])
    #extra = np.array([66/256, 75/256, 98/256, 1])
    #extra = np.array([3/256, 34/256, 71/256, 1])
    extra = np.array([0, 0, 0, 1])
    newcolors[-10:, :] = extra
    newcmap = ListedColormap(newcolors)
    return newcmap

def plot_GW_test_pvals(pvals, labels, title):
    assert len(pvals)==len(labels), 'Wrong pvals and labels dimensions.'

    #plt.rc('text', usetex=True)
    plt.rc('axes', labelsize=FONTSIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONTSIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONTSIZE)    # fontsize of the tick labels
    plt.rc('axes', titlesize=FONTSIZE+1)  # fontsize of the figure title

    fig = plt.figure(figsize=[6, 6])
    ax = plt.axes([.27, .22, .7, .7])

    data = np.float32(pvals)

    # Colormap with discontinuous limit
    #cmap = cm.get_cmap('GnBu', 256)
    #cmap = get_nbeatsx_cmap()
    cmap = get_epftoolbox_cmap()
    mappable = plt.imshow(data, cmap=cmap, vmin=0, vmax=0.1)

    ticklabels = labels #[r'$\textrm{' + e + '}$' for e in labels]
    plt.xticks(range(len(labels)), ticklabels, rotation=90., fontsize=FONTSIZE)
    plt.yticks(range(len(labels)), ticklabels, fontsize=FONTSIZE)

    plt.plot(list(range(len(labels))), 
             list(range(len(labels))), 'wx', c='white', markersize=FONTSIZE)
    plt.title(f'{title}', fontweight='bold', fontsize=FONTSIZE)

    # Turn spines off and create black grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=1.5)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2) #0.05
    plt.colorbar(mappable, cax=cax)

    #fig.tight_layout()
    title = title.replace(" ", "_")
    title = title.replace(",", "")
    title = title.replace("(", "")
    title = title.replace(")", "")
    plt.savefig(f'./results/pvals/pvals_{title}.pdf', bbox_inches='tight')
    plt.show()

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

def rmae(y: np.ndarray,
         y_hat1: np.ndarray, y_hat2: np.ndarray,
         weights=None):
    """Calculates Relative Mean Absolute Error.

    The relative mean absolute error of two forecasts.
    A number smaller than one implies that the forecast in the
    numerator is better than the forecast in the denominator.

    Parameters
    ----------
    y: numpy array
      actual test values
    y_hat1: numpy array
      predicted values of first model
    y_hat2: numpy array
      predicted values of second model
    weights: numpy array
      weights for weigted average
    freq: int
      frequency of the y series, it will determine the
      seasonal naive benchmark

    Return
    ------
    scalar: rMAE
    """
    numerator = mae(y=y, y_hat=y_hat1, weights=weights)
    denominator = mae(y=y, y_hat=y_hat2, weights=weights)
    rmae = numerator/denominator
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