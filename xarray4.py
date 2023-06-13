#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:50:57 2022

@author: mohammadbasiruddin
"""
##xarray scatter plot use pattern
import xarray as xr
ds1=xr.open_dataset('2000monthly-surft-prec.nc')
#print(ds)
ds1.plot.scatter(x="t2m",y="lsp", hue="latitude", cmap='viridis')

