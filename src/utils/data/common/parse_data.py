import numpy as np
import pandas as pd

def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    """
    Filter values array by group indices and clean it from NaNs.
    :param values: Values to filter.
    :param groups: Timeseries groups.
    :param group_name: Group name to filter by.
    :return: Filtered and cleaned timeseries.
    """
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])

def group_str_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    """
    Filter values array by group indices and clean it from NaNs.
    :param values: Values to filter.
    :param groups: Timeseries groups.
    :param group_name: Group name to filter by.
    :return: Filtered and cleaned timeseries.
    """
    return np.array([v for v in values[groups == group_name]])

def wide_to_long_df(values, ids, dss):
    y = np.concatenate(values)
    unique_id = np.concatenate([np.array([unique_id] * len(ds)) \
                                    for unique_id, ds in zip(ids, dss)])
    ds = np.concatenate(dss)
    long_df = pd.DataFrame({'unique_id': unique_id,
                            'ds': ds,
                            'y': y})
    return long_df
