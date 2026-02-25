# dataset.py
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np

class CodespaceWeatherDataset(Dataset):
    def __init__(self, nc_file_path, seq_length, predict_ahead=1):
        # Lazy loading to prevent Codespace RAM crashes
        self.ds = xr.open_dataset(nc_file_path, engine='netcdf4', chunks={})
        self.variables = ['tp', 'd2m', 'u10', 'v10', 'sp']
        
        # Find the time dimension (it might have different names)
        time_dim = None
        for dim in self.ds.dims:
            if 'time' in dim.lower():
                time_dim = dim
                break
        
        if not time_dim:
            raise ValueError(f"No time dimension found. Available dims: {list(self.ds.dims.keys())}")
        
        self.time_dim = time_dim
        self.seq_length = seq_length
        self.predict_ahead = predict_ahead
        self.total_time_steps = self.ds.dims[time_dim]
        self.num_samples = self.total_time_steps - self.seq_length - self.predict_ahead + 1

        # Calculate min/max for normalization once during init
        self.mins = {var: self.ds[var].min().compute().item() for var in self.variables}
        self.maxs = {var: self.ds[var].max().compute().item() for var in self.variables}

    def __len__(self):
        return self.num_samples

    def _normalize(self, data_array, var_name):
        min_val = self.mins[var_name]
        max_val = self.maxs[var_name]
        range_val = max_val - min_val if max_val - min_val != 0 else 1.0
        return (data_array - min_val) / range_val

    def __getitem__(self, idx):
        # 1. Slice time window directly from disk
        past_window = self.ds.isel({self.time_dim: slice(idx, idx + self.seq_length)})
        target_step = self.ds.isel({self.time_dim: idx + self.seq_length + self.predict_ahead - 1})
        
        X_list, Y_list = [], []
        
        # 2. Extract and normalize
        for var in self.variables:
            x_vals = past_window[var].values 
            y_vals = target_step[var].values
            
            # Fill NaNs with 0.0 (equivalent to your np.nan_to_num step)
            x_vals = np.nan_to_num(x_vals, nan=0.0)
            y_vals = np.nan_to_num(y_vals, nan=0.0)
            
            X_list.append(self._normalize(x_vals, var))
            Y_list.append(self._normalize(y_vals, var))
            
        # 3. Stack into (C, T, H, W)
        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)
        
        # 4. Convert to tensors and permute to (T, C, H, W)
        X_tensor = torch.tensor(X, dtype=torch.float32).permute(1, 0, 2, 3)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        
        return X_tensor, Y_tensor