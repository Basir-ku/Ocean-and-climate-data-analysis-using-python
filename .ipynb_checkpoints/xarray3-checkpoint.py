#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:51:57 2022

@author: mohammadbasiruddin
"""
#ploting example: plotting sea level temp. in Hamburg

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

##customizing x array plot.comparing mean temperature in Hamburg and paris

import matplotlib.pyplot as plt
tham=ds.t.sel(level=100.0, latitude="53.55",longitude=10.0, method="nearest")
tpar=ds.t.sel(level=100.0, latitude="48.86",longitude=2.35, method="nearest")
#Paris location is 48.86N and 2.35 E
tham.plot(color='blue', marker='o', label="Hamburg")
tpar.plot(color='red',label="Paris")
plt.legend()
plt.title('Sea level mean monthly temperature in 2000')

import datetime
date1 = datetime.date(year=2000,day=1,month=5)
tsea=ds.t.sel(level=1000.0, time=date1, method="nearest")
tsea.plot(add_colorbar=True, cmap="coolwarm", fontsize=16)
