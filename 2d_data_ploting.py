#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:31:55 2022

@author: mohammadbasiruddin
"""

import numpy as np
import matplotlib.pyplot as plt

lon=np.linspace(0,360,361)
lat=np.linspace(-90,90,181)
X,Y=np.meshgrid(lon,lat)

pi=np.pi
T=270*(np.cos(Y*pi/180)**2)*(np.cos(X*pi/180)**2)+100.0

plt.pcolormesh(X,Y,T,cmap='coolwarm',vmin=100,vmax=350)
plt.colorbar()
plt.contourf(X,Y,T,levels=5)
plt.grid()
plt.xlabel('langitude[ degree]', fontsize=15)
plt.ylabel('latitude [degree]', fontsize=15)
plt.title('temp. 180 pha on a fantasy palnet')
plt.show()

