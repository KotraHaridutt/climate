#!/usr/bin/env python3
import xarray as xr
import sys

print("Loading data files...")
try:
    ds_accum = xr.open_dataset('../testing/data_stream-oper_stepType-accum.nc')
    print(f"✓ Loaded accumulated file")
    print(f"  Dimensions: {dict(ds_accum.dims)}")
    print(f"  Coordinates: {list(ds_accum.coords)}")
    print(f"  Variables: {list(ds_accum.data_vars)}")
except Exception as e:
    print(f"✗ Error loading accum file: {e}")
    sys.exit(1)

try:
    ds_instant = xr.open_dataset('../testing/data_stream-oper_stepType-instant.nc')
    print(f"✓ Loaded instantaneous file")
    print(f"  Dimensions: {dict(ds_instant.dims)}")
    print(f"  Coordinates: {list(ds_instant.coords)}")
    print(f"  Variables: {list(ds_instant.data_vars)}")
except Exception as e:
    print(f"✗ Error loading instant file: {e}")
    sys.exit(1)

# Try to identify the time dimension
time_dim = None
for dim in ds_accum.dims:
    if 'time' in dim.lower():
        time_dim = dim
        break
if not time_dim and len(ds_accum.dims) > 0:
    time_dim = list(ds_accum.dims.keys())[0]

print(f"\nDetected time dimension: {time_dim}")

try:
    # Merge using compatible_align
    merged = xr.merge([ds_accum, ds_instant], join='override')
    
    # Save as telangana_weather_data.nc
    merged.to_netcdf('../testing/telangana_weather_test_data.nc')
    print("\n✓ Successfully merged and saved to telangana_weather_data.nc")
    
except Exception as e:
    print(f"\n✗ Error during merge: {e}")
    print("\nAttempting alternative merge strategy...")
    # If standard merge fails, try combining along existing dimension
    try:
        for var in ds_instant.data_vars:
            ds_accum[var] = ds_instant[var]
        ds_accum.to_netcdf('telangana_weather_data.nc')
        print("✓ Successfully merged using variable assignment")
    except Exception as e2:
        print(f"✗ Alternative merge also failed: {e2}")
        sys.exit(1)

ds_accum.close()
ds_instant.close()
