#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 23:22:12 2022

@author: mohammadbasiruddin
"""

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
fname='2000monthly-meant.nc'
ds=nc.Dataset(fname)

## Location of Hamburg is 53.55N, 10E
# Find in the data latitude and longitude closest to Hamburg
lat=ds["latitude"][:]
lon=ds["longitude"][:]
ilat=np.argmin(np.abs(lat-53.55))
ilon=np.argmin(np.abs(lon-10.00))

#As we learned before, the order of indices is time, level, latitude, longitude
#near surface values correspond to the last level index
#so the surface time series for Hamburg is
t_ham=ds["t"][:,-1,ilat,ilon]
#plt.plot(ds["time"][:],t_ham)
from cftime import num2date
date=num2date(ds["time"][:],units=ds["time"].units,
calendar=ds["time"].calendar)
plt.plot_date(date,t_ham, 'b-o')