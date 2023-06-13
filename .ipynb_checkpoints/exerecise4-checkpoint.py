#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:44:55 2022

@author: mohammadbasiruddin
"""

import numpy as np
import matplotlib.pyplot as plt

#varibale and constant
pi=np.pi

#generating lon and lat

lon=np.linspace(0,360,361)
lat=np.linspace(-90,90,181)
X,Y=np.meshgrid(lon,lat)

#equations
U=-10*(np.sin(2*Y*pi/180))*(np.cos(X*pi/180)**2)
V=10*(np.cos(Y*pi/180)**2)*(np.sin(2*X*pi/180))

#compute the zonal average speed
U_ave=np.average(U,1) 
V_ave=np.average(V,1)
speed= np.sqrt(U_ave**2+V_ave**2)

#ploting the wind speed as a fun of lat
plt.plot(speed,lat)
plt.xlabel('speed')
plt.ylabel('latitude')
plt.title('xonal average windspeed')