import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

#read file
fname='2000monthly-surft-prec.nc'
ds = xr.open_dataset(fname)

#select 25 X 25 box
t=ds.t2m
i1=np.where(t.latitude<48+12.5)[0][0]
i2=np.where(t.latitude>48-12.5)[0][-1]
i3=np.where(t.longitude<25)[0][-1]

temp=t.isel(longitude=slice(0,i3+1),latitude=slice(i1,i2+1))
prec=ds.lsp.isel(longitude=slice(0,i3+1),latitude=slice(i1,i2+1))
#Temperature averaged over time
fig=plt.figure()
ax=plt.gca(projection=ccrs.PlateCarree())
ax.set_extent([-10, 30, 30, 65])

t_av=temp.mean(dim="time")
t_av.plot()

ax.coastlines()
ax.set_title("Temperature averaged over time")
ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='gray')
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.RIVERS, linewidth=0.6, edgecolor='red')
gl=ax.gridlines(draw_labels=True,
linewidth=0.5, color='black', linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False