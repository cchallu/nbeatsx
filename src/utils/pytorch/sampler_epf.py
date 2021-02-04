import numpy as np
import pandas as pd
import random
import torch as t

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class TimeseriesDataset(Dataset):
    def __init__(self,
                 model:str,
                 ts_data: list,
                 static_data: list,
                 meta_data: list,
                 offset:int,
                 window_sampling_limit: int, 
                 input_size: int,
                 output_size: int,
                 idx_to_sample_freq:int,
                 batch_size: int):
        """
        """
        self.model = model
        self.window_sampling_limit = window_sampling_limit
        self.input_size = input_size
        self.output_size = output_size
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.idx_to_sample_freq = idx_to_sample_freq
        self.offset = offset

        self.n_series = len(ts_data)
        self.max_len = max([len(ts['y']) for ts in ts_data])
        self.n_channels = len(ts_data[0].values())

        self.time_series_tensor, static_data, self.len_series = self.create_tensor(ts_data, static_data)
        self.ts_windows = self.create_windows_tensor(self.time_series_tensor, input_size, output_size)
        self.n_windows = len(self.ts_windows)
        self.static_data = static_data.repeat(int(self.n_windows/self.n_series), 1)

        self.update_offset(offset)
        self._is_train = True
        #random.seed(1)

    def update_offset(self, offset):
        self.offset = offset
        self.ts_windows = self.create_windows_tensor(self.time_series_tensor, self.input_size, self.output_size)
        self.n_windows = len(self.ts_windows)

    def train(self):
        self._is_train = True

    def eval(self):
        self._is_train = False

    def get_meta_data_var(self, var):
        var_values = [x[var] for x in self.meta_data]
        return var_values

    def create_tensor(self, ts_data, static_data):
        ts_tensor = np.zeros((self.n_series, self.n_channels + 1, self.max_len)) # + 1 for the mask of ones
        static_tensor = np.zeros((self.n_series, len(static_data[0])))

        len_series = []
        for idx in range(self.n_series):
            ts_idx = np.array(list(ts_data[idx].values()))
            ts_tensor[idx, :-1, -ts_idx.shape[1]:] = ts_idx
            ts_tensor[idx, -1, -ts_idx.shape[1]:] = 1 # Mask
            static_tensor[idx, :] = list(static_data[idx].values())
            len_series.append(ts_idx.shape[1])
        
        ts_tensor = t.Tensor(ts_tensor)
        static_tensor = t.Tensor(static_tensor)

        return ts_tensor, static_tensor, np.array(len_series)

    def filter_time_series_tensor(self, tensor):
        """
        Comment here
        """
        last_ds = self.max_len - self.offset + self.output_size
        first_ds = last_ds - self.window_sampling_limit - self.output_size
        filter_tensor = tensor[:, :, first_ds:last_ds]
        right_padding = max(last_ds - self.max_len, 0)
        return filter_tensor, right_padding

    def create_windows_tensor(self, tensor, input_size, output_size):
        """
        Comment here
        """
        tensor, right_padding = self.filter_time_series_tensor(tensor)
        _, c, _ = tensor.size()

        padder = t.nn.ConstantPad1d(padding=(input_size-1, right_padding), value=0)
        tensor = padder(tensor)

        tensor[:, 0, -output_size:] = 0
        tensor[:, -1, -output_size:] = 0

        windows = tensor.unfold(dimension=-1, size=input_size + output_size, step=1)
        windows = windows.permute(2,0,1,3)
        windows = windows.reshape(-1, c, input_size + output_size)
        return windows

    def __len__(self):
        return len(self.len_series)

    def __iter__(self):
        while True:
            if self._is_train:
                sampled_ts_indices = np.random.randint(low=0, high=self.n_windows, size=self.batch_size)
            else:
                sampled_ts_indices = list(range(self.n_windows-self.n_series, self.n_windows))

            #print(sampled_ts_indices)

            batch = self.__get_item__(sampled_ts_indices)

            #print(batch)

            yield batch

    def __get_item__(self, index):
        if self.model == 'nbeats':
            return self.nbeats_batch(index)
        elif self.model == 'esrnn':
            assert 1<0, 'hacer esrnn'
        else:
            assert 1<0, 'error'

    def nbeats_batch(self, index):

        windows = self.ts_windows[index]
        static_data = self.static_data[index]

        insample_y = windows[:, 0, :self.input_size]
        insample_x_t = windows[:, 1:-1, :self.input_size]
        insample_mask = windows[:, -1, :self.input_size]

        outsample_y = windows[:, 0, self.input_size:]
        outsample_x_t = windows[:, 1:-1, self.input_size:]
        outsample_mask = windows[:, -1, self.input_size:]

        batch = {'insample_y':insample_y, 'insample_x_t':insample_x_t, 'insample_mask':insample_mask,
                  'outsample_y':outsample_y, 'outsample_x_t':outsample_x_t, 'outsample_mask':outsample_mask,
                  'static_data':static_data}

        return batch
