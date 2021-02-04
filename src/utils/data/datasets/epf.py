"""
EPF Dataset
"""
import logging
import os
import pickle
from six.moves import urllib
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from datetime import timedelta

import numpy as np
import pandas as pd
import patoolib
from tqdm import tqdm

from src.utils.data.common.http_utils import download, url_file_name
from src.utils.data.common.parse_data import *
from src.utils.data.common.settings import DATASETS_PATH

#SOURCE_URL = 'https://sandbox.zenodo.org/api/files/da5b2c6f-8418-4550-a7d0-7f2497b40f1b/'
MARKETS = ['NP','PJM', 'FR', 'BE', 'DE']

TEST_DATE = {'NP': '2016-12-27',
             'PJM':'2016-12-27',
             'BE':'2015-01-04',
             'FR': '2015-01-04',
             'DE':'2016-01-04'}

# MARKET_URLS ={market: (SOURCE_URL + f'{market}.csv') \
#                                 for market in ['PJM', 'NP', 'FR', 'BE', 'DE']}

MARKET_PATHS ={market:os.path.join(DATASETS_PATH + f'epf/{market}') \
                                for market in ['PJM', 'NP', 'FR', 'BE', 'DE']}

def load_epf(market, first_date_test, days_in_test):
    assert market in MARKETS, \
        f'Market {market} not available'
    
    data_dir = MARKET_PATHS[market] + '.pkl'
    if not os.path.exists(data_dir):
        df = EPFDataset.load(market).df
        with open(data_dir, "wb") as f:
            pickle.dump(df, f)
    df = pickle.load(open(data_dir, "rb" ))

    df['ds'] = pd.to_datetime(df['ds'])

    df['week_day'] = df['ds'].dt.dayofweek
    dummies = pd.get_dummies(df['week_day'],prefix='day')
    df = pd.concat((df,dummies), axis=1)

    dummies_cols = [col for col in df if col.startswith('day')]

    y_insample_df = df[df['ds']<first_date_test].reset_index(drop=True)[['unique_id','ds','y']]
    X_insample_df = df[df['ds']<first_date_test].reset_index(drop=True)[['unique_id','ds', 'Exogenous1', 'Exogenous2', 'week_day'] + dummies_cols]

    last_date_test = pd.to_datetime(first_date_test)+ timedelta(days=days_in_test)

    y_outsample_df = df[(df['ds']>=first_date_test) &
                        (df['ds']<last_date_test)].reset_index(drop=True)[['unique_id','ds','y']]
    X_outsample_df = df[(df['ds']>=first_date_test) &
                        (df['ds']<last_date_test)].reset_index(drop=True)[['unique_id','ds', 'Exogenous1', 'Exogenous2', 'week_day'] + dummies_cols]

    X_s_df = None

    return y_insample_df, X_insample_df, y_outsample_df, X_outsample_df, X_s_df

@dataclass()
class EPFDataset:
    df: pd.DataFrame

    @staticmethod
    def load(market: str) -> 'EPFDataset':
        assert market in ['PJM', 'NP', 'FR', 'BE', 'DE']

        EPFDataset.download()
        
        df = pd.read_csv(MARKET_PATHS[market]+'.csv')
        
        columns = ['ds','y']
        n_exogeneous_inputs = len(df.columns) - 2

        for n_ex in range(1, n_exogeneous_inputs + 1):
            columns.append('Exogenous' + str(n_ex))

        df.columns = columns

        df['unique_id'] = market
        
        dataset = EPFDataset(df=df)
        return dataset

    @staticmethod
    def download() -> None:
        for market in MARKETS:
            if not os.path.exists(MARKET_PATHS[market]+'.csv'):
                download(MARKET_URLS[market], MARKET_PATHS[market]+'.csv')