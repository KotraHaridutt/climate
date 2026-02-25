# download_data.py
import cdsapi

c = cdsapi.Client()
area = [20, 77, 15, 82] 

print("Starting download. This might take a few minutes...")
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'total_precipitation',
            '2m_dewpoint_temperature', # Fixed: Surface moisture proxy
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'surface_pressure',
        ],
        'year': '2023',
        'month': ['06', '07', '08'], 
        'day': [str(i).zfill(2) for i in range(1, 32)],
        'time': [f"{str(i).zfill(2)}:00" for i in range(24)],
        'area': area,
    },
    'telangana_weather_data.nc'
)
print("Download complete!")