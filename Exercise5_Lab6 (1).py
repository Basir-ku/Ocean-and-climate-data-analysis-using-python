# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 13:45:22 2021

@author: mohammadbasiruddin
"""

""

import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
#Exercise 5

fname = 'May2000-uvt.nc'
ds = xr.open_dataset(fname)

#choose a latitude lambda l1=53.5511 l2= -53.5511 

#Plot histograms for the zonal wind at N and S and pressure level 500mb.

#for the north 
u1 = ds.u.sel(latitude=40,level=500,  method="nearest") #zonal component
U1   = u1.to_dataframe()
U11  = U1.reset_index(level='longitude')


#calculate mean and std
mean1 = np.mean(U11.u)

std1 = np.std(U11.u, ddof=1)

#plot 
sns.set(style="darkgrid") 
ax1   = sns.histplot(data=U11.u, stat="density")


sns.kdeplot(U11.u,ax=ax1,color="green", label='mean =%3.2f\n std=%3.2f' %(mean1,std1))
plt.legend()
plt.title("zonal wind at 40N")
plt.savefig('Ex5_north',dpi=1000)

######for the south 
u2 = ds.u.sel(latitude= -40,level=500, method="nearest")

U2   = u2.to_dataframe()
U22  = U2.reset_index(level='longitude')

#calculate mean and std

mean2 = np.mean(U22.u)
std2 = np.std(U22.u, ddof=1)

#plot for the south

sns.set(style="darkgrid") 
ax2   = sns.histplot(data=U22.u, stat="density")

sns.kdeplot(U22.u,ax=ax2,color="green", label='mean =%3.2f\n std=%3.2f' %(mean2,std2))
plt.title("zonal wind at 40S")
plt.legend()
plt.savefig('Ex5_south',dpi=1000)


#compare 
to,pv=scipy.stats.ttest_ind(mean1,mean2)
print('The Samples do not have the same mean', to)





        