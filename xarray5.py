#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:44:04 2022

@author: mohammadbasiruddin
"""

import xarray as xr
ds1=xr.open_dataset('2000monthly-surft-prec.nc')
ax1=tsea.plot.contour(levels=10, colors="blue", linewidths=0.5,linestyles=’solid’)
plt.clabel(ax1, inline=1, fontsize=10)