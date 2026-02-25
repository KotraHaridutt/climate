# main.py
from torch.utils.data import DataLoader
from dataset import CodespaceWeatherDataset
import os

def main():
    # Auto-merge if needed
    if not os.path.exists('telangana_weather_data.nc'):
        if os.path.exists('data_stream-oper_stepType-accum.nc'):
            print("Merging extracted data files...")
            import xarray as xr
            ds_accum = xr.open_dataset('data_stream-oper_stepType-accum.nc')
            ds_instant = xr.open_dataset('data_stream-oper_stepType-instant.nc')
            merged = xr.merge([ds_accum, ds_instant], join='override')
            merged.to_netcdf('telangana_weather_data.nc')
            ds_accum.close()
            ds_instant.close()
            print("✓ Data merged successfully!")
        else:
            print("✗ Error: telangana_weather_data.nc or extracted files not found")
            return
    
    print("Initializing Dataset...")
    # Parameters
    sequence_length = 24 # 24 hours of history
    batch_size = 4       # Keep it small for Codespaces
    
    # Create the dataset instance
    dataset = CodespaceWeatherDataset('telangana_weather_data.nc', seq_length=sequence_length)
    
    # Set num_workers=0 to prevent multiprocessing memory spikes
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    print(f"Total batches available: {len(dataloader)}")
    
    # Test a single batch to verify shapes
    for batch_X, batch_Y in dataloader:
        print(f"Success! Input X shape (B, T, C, H, W): {batch_X.shape}") 
        print(f"Success! Target Y shape (B, C, H, W): {batch_Y.shape}") 
        break # Break after one iteration just to test

if __name__ == "__main__":
    main()