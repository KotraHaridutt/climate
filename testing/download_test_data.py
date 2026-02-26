# download_test_data.py
import cdsapi

c = cdsapi.Client()
area = [20, 77, 15, 82] 

print("Downloading September test data...")
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'total_precipitation',
            '2m_dewpoint_temperature', 
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'surface_pressure',
        ],
        'year': '2023',
        'month': '09', 
        'day': [str(i).zfill(2) for i in range(1, 8)], # First week of Sept
        'time': [f"{str(i).zfill(2)}:00" for i in range(24)],
        'area': area,
    },
    'telangana_test_data.nc'
)
print("Test data downloaded!")