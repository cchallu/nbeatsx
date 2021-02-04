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

        self.time_series_tensor, self.static_data, self.len_series = self.create_tensor(ts_data, static_data)
        self.ts_windows = self.create_windows_tensor(self.time_series_tensor, input_size, output_size)
        self.n_windows = len(self.ts_windows)
        self.mask_windows = self.create_windows_tensor(t.ones(self.n_series, 1, self.max_len), input_size, output_size)
        self.mask_windows = self.mask_windows.squeeze(1)
        self.static_data_windows = self.static_data.repeat(int(self.n_windows/self.n_series), 1)

        self.update_offset(offset)
        self._is_train = True
        #random.seed(1)

    def update_offset(self, offset):
        self.offset = offset
        self.ts_windows = self.create_windows_tensor(self.time_series_tensor, self.input_size, self.output_size)
        self.n_windows = len(self.ts_windows)
        self.mask_windows = self.create_windows_tensor(t.ones(self.n_series, 1, self.max_len), self.input_size, self.output_size)
        self.mask_windows = self.mask_windows.squeeze(1)
        self.static_data_windows = self.static_data.repeat(int(self.n_windows/self.n_series), 1)

    def train(self):
        self._is_train = True

    def eval(self):
        self._is_train = False

    def get_meta_data_var(self, var):
        var_values = [x[var] for x in self.meta_data]
        return var_values

    def create_tensor(self, ts_data, static_data):
        ts_tensor = np.zeros((self.n_series, self.n_channels, self.max_len))
        static_tensor = np.zeros((self.n_series, len(static_data[0])))

        len_series = []
        for idx in range(self.n_series):
            ts_idx = np.array(list(ts_data[idx].values()))
            ts_tensor[idx, :, -ts_idx.shape[1]:] = ts_idx
            static_tensor[idx, :] = list(static_data[idx].values())
            len_series.append(ts_idx.shape[1])
        
        ts_tensor = t.Tensor(ts_tensor)
        static_tensor = t.Tensor(static_tensor)

        return ts_tensor, static_tensor, np.array(len_series)

    def filter_time_series_tensor(self, tensor):
        """
        Comment here
        """
        last_ds = self.max_len - self.offset
        first_ds = last_ds - self.window_sampling_limit
        filter_tensor = tensor[:, :, first_ds:last_ds]

        return filter_tensor

    def create_windows_tensor(self, tensor, input_size, output_size):
        """
        Comment here
        """
        tensor = self.filter_time_series_tensor(tensor)
        _, c, _ = tensor.size()

        padder = t.nn.ConstantPad1d(padding=(input_size-1, output_size), value=0)
        tensor = padder(tensor)

        windows = tensor.unfold(dimension=-1, size=input_size + output_size, step=1)
        windows = windows.permute(2,0,1,3)
        windows = windows.reshape(-1, c, input_size + output_size)
        return windows

    def permute_windows_tensor(self):
        shuffle = t.randperm(self.n_windows)
        self.ts_windows = self.ts_windows[shuffle]
        self.mask_windows = self.mask_windows[shuffle]
        self.static_data_windows = self.static_data_windows[shuffle]

    def __len__(self):
        return len(self.len_series)

    def __iter__(self):
        self.counter = 0
        self.permute_windows_tensor() #TODO: definir donde meter esto
        while True:
            if self._is_train:
                start_index = self.counter * self.batch_size
                end_index = start_index + self.batch_size
                batch = self.__get_item__(start_index, end_index)
                self.counter += 1
                if end_index > self.n_windows:
                    self.counter = 0
                    self.permute_windows_tensor()
            else:
                assert 1<0, 'TODO'

            yield batch

    def __get_item__(self, start_index, end_index):
        if self.model == 'nbeats':
            return self.nbeats_batch(start_index, end_index)
        elif self.model == 'esrnn':
            assert 1<0, 'hacer esrnn'
        else:
            assert 1<0, 'error'

    def nbeats_batch(self, start_index, end_index):

        windows = self.ts_windows[start_index:end_index]
        mask = self.mask_windows[start_index:end_index]
        static_data = self.static_data_windows[start_index:end_index]

        insample_y = windows[:, 0, :self.input_size]
        insample_x_t = windows[:, 1:, :self.input_size]
        insample_mask = mask[:, :self.input_size]

        outsample_y = windows[:, 0, self.input_size:]
        outsample_x_t = windows[:, 1:, self.input_size:]
        outsample_mask = mask[:, self.input_size:]

        batch = {'insample_y':insample_y, 'insample_x_t':insample_x_t, 'insample_mask':insample_mask,
                  'outsample_y':outsample_y, 'outsample_x_t':outsample_x_t, 'outsample_mask':outsample_mask,
                  'static_data':static_data}

        return batch
