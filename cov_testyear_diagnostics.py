import xarray as xr
import numpy as np

ds = xr.open_dataset("Data/CovsDaily/daily_gridded_CA_2024.nc")

# Monthly block ID: year*12 + month
months = (ds['time'].dt.month - 1) // 2
ds = ds.assign_coords(block=months)
block_cov = ds.groupby("block").mean(["time", "cell"]).rename({v: f"{v}_mean" for v in ds.data_vars})
block_var = ds.groupby("block").var(["time", "cell"]).rename({v: f"{v}_var" for v in ds.data_vars})
block_features = xr.merge([block_cov, block_var])

import statsmodels.api as sm
import matplotlib.pyplot as plt

for name,data in block_features.data_vars.items():
    sm.graphics.tsa.plot_acf(data.values, lags=5)
    plt.title(name)
    plt.savefig(f'acf_plots/{name}_acf.png')
