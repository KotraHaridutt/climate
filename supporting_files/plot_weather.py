# plot_weather.py
import xarray as xr
import matplotlib.pyplot as plt

# Load the data
ds = xr.open_dataset('../data_stream-oper_stepType-accum.nc')

# Find the specific hour with the absolute maximum rainfall in the entire dataset
max_rain_idx = ds['tp'].argmax(dim=['latitude', 'longitude'])  # This gives us the index of the time step with the heaviest rainfall
# Extract that specific time slice
heaviest_rain_slice = ds.isel(valid_time=max_rain_idx)    

# Convert from meters to millimeters
rain_data_mm = heaviest_rain_slice['tp'] * 1000

# Create the plot
plt.figure(figsize=(8, 6))
# Plot the 2D grid. cmap='Blues' makes it look like rain!
plot = plt.imshow(rain_data_mm, cmap='Blues', origin='lower') 

# Add colorbar and labels
plt.colorbar(plot, label='Total Precipitation (mm/hour)')
plt.title(f"Heaviest Rainfall Event in Dataset\nTime: {heaviest_rain_slice.valid_time.values}")
plt.xlabel("Longitude Grid Steps")
plt.ylabel("Latitude Grid Steps")

# Save the image to your Codespace directory
plt.savefig("heaviest_rain_map.png", dpi=300, bbox_inches='tight')
print("Saved 'heaviest_rain_map.png'! Click on it in your file explorer to view.")