# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:14:52 2021

@author: lujua
"""


import xarray as xr

import matplotlib.pyplot as plt

import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import scipy.stats
#Exercise 2

fname = 'May2000-uvt.nc'
ds = xr.open_dataset(fname)

u = ds.u.sel(level=1000, method="nearest") #surface zonal wind 
v = ds.v.sel(level=1000,  method="nearest")

x10=ds.longitude
y10=ds.latitude

qsx=6 # step along x axis
qsy=3 # step along y axis
x10=x10.isel(longitude=slice(0,144,qsx))
y10=y10.isel(latitude=slice(0,73,qsy))
u10=u.isel(longitude=slice(0,144,qsx),latitude=slice(0,73,qsy))
v10=v.isel(longitude=slice(0,144,qsx),latitude=slice(0,73,qsy))





av_u10 = u10.mean(dim="time")
av_v10 = v10.mean(dim="time")


#calculate windspeed
wind_speed_10 = np.sqrt(av_u10**2 + av_v10**2)
               

location = u.isel(time=0)

np.cov(u[:,0,0],v[:,0,0])
scipy.stats.pearsonr(u[:,0,0], v[:,0,0])

#plot
for n in range(len(ds.latitude)):
    for k in range(len(ds.longitude)):
        location[n,k]=scipy.stats.pearsonr(u[:,n,k],v[:,n,k])[0]

ax = plt.gca(projection=ccrs.PlateCarree())
ax.coastlines()
#qv= ax.quiver(ds.longitude,ds.latitude, u,v, color= "red", scale=1000)
location.plot(ax=ax )#, cmap = "coolwarm") 


qv = ax.quiver(x10, y10,av_u10,av_v10, angles="uv", transform=ccrs.PlateCarree(), color='green')
q_typ=wind_speed_10.max()
Qkey = ax.quiverkey(qv ,0.3, 1.12, q_typ, '%.1f m/s' % q_typ )

plt.savefig('exercise2',dpi=1000)

