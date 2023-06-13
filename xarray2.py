#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:40:38 2022

@author: mohammadbasiruddin
"""

import xarray as xr
fname="2000monthly-meant.nc"
ds=xr.open_dataset(fname)
print(ds)
tham=ds.t.sel(level=100.0,latitude='53.55',longitude=10.0, method='nearest')
tham.plot()

## customizing the plot using Matplotlip

import matplotlib.pyplot as plt
plt.grid()
plt.xlabel('Date')
plt.title('monthly mean sea level temperature in Hamburg')