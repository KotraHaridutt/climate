#!/usr/bin/env python3
import xarray as xr

print("=" * 60)
print("Analyzing both files...")
print("=" * 60)

try:
    ds_accum = xr.open_dataset('data_stream-oper_stepType-accum.nc')
    print("\n📊 ACCUMULATED FILE:")
    print(f"Dimensions: {dict(ds_accum.dims)}")
    print(f"Coordinates: {dict(ds_accum.coords)}")
    print(f"Variables: {list(ds_accum.data_vars)}")
    print(f"\nFull structure:\n{ds_accum}")
    ds_accum.close()
except Exception as e:
    print(f"Error reading accum file: {e}")

print("\n" + "=" * 60)

try:
    ds_instant = xr.open_dataset('data_stream-oper_stepType-instant.nc')
    print("\n📊 INSTANTANEOUS FILE:")
    print(f"Dimensions: {dict(ds_instant.dims)}")
    print(f"Coordinates: {dict(ds_instant.coords)}")
    print(f"Variables: {list(ds_instant.data_vars)}")
    print(f"\nFull structure:\n{ds_instant}")
    ds_instant.close()
except Exception as e:
    print(f"Error reading instant file: {e}")
