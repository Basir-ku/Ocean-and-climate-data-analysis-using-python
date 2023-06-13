#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 00:05:02 2022

@author: mohammadbasiruddin
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt

#first step, read files
xfile=open("kdvx.dat","rb")
x=np.load(xfile)
tfile=open("kdvt.dat","rb")
t=np.load(tfile)
ufile=open("kdvsol.dat","rb")
u=np.load(ufile)

#now, produceing Hovmoller diagram
plt.contourf(x,t,u)
plt.colorbar()
plt.xlabel('x') 
plt.ylabel('t') 
plt.title("KDV equation")

plt.savefig('kdv_figure')