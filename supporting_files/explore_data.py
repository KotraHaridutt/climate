# explore_data.py
import xarray as xr
import numpy as np

# 1. Open the dataset
ds = xr.open_dataset('../data_stream-oper_stepType-accum.nc')

print("=== DATASET OVERVIEW ===")
# This prints the dimensions, coordinates, and variables
print(ds) 

print("\n=== PRECIPITATION STATISTICS ===")
# ERA5 stores Total Precipitation (tp) in meters. 
# Let's convert it to millimeters (mm) so it makes intuitive sense.
precip_mm = ds['tp'] * 1000 

print(f"Maximum rainfall in a single hour: {precip_mm.max().item():.2f} mm")
print(f"Average rainfall: {precip_mm.mean().item():.4f} mm")

# To define an "extreme" event, we usually look at the 99th percentile
percentile_99 = precip_mm.quantile(0.99).item()
print(f"99th Percentile rainfall: {percentile_99:.2f} mm")
print("(This means 99% of the time, rainfall is below this number. Anything above this is your 'extreme' target!)")