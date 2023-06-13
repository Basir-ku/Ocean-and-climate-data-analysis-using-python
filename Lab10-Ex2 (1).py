# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:06:09 2021

@author: mohammadbasiruddin
"""

#Lab10-Ex2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error as mse, r2_score as rsq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

ds = pd.read_csv('elnino-80-98.csv')

data_in = ds[['Zonal_Winds','Meridional_Winds','Air_Temp','SST']]
data_in = data_in.interpolate()
data_out = ds['SST3m']
data_out= data_out.interpolate()

# shallow learning model: Random Forest
# split the data into training and testing sets

train_in, test_in, train_out, test_out = train_test_split(data_in, data_out, test_size = 0.25)
rf = RandomForestRegressor(n_estimators=30)
rf.fit(train_in,train_out)
Prediction = rf.predict(test_in)
Training = rf.predict(train_in)

#plot 
plt.plot(test_out,Prediction,linestyle="",marker="x")
plt.xlabel('Prediction')
plt.ylabel('test output')
plt.savefig('test_out-prediction',dpi=1000)
#plt.plot(test_in,test_out)
print("RF Training: %3.1f, MSE: %3.3f " %( rsq(train_out, Training), mse(train_out,Training)))
print("RF Testing: %3.1f, MSE: %3.3f " %( rsq(test_out, Prediction), mse(test_out,Prediction) ))

print(np.corrcoef(test_out,Prediction))