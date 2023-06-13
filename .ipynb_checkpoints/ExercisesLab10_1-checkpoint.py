#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:53:19 2022

@author: svenja
"""

#get packages
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error as mse, r2_score as rsq

from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import tensorflow as tf
import tensorflow.keras as keras

#%% Exercises 1

#define function for time series
def ts(a,b,w,phi,t): #to generate time series
    nois = np.random.normal(loc=1,scale=5, size=201)
    return a + b * np.sin(w*t + phi) + 0.3 * nois

#compute time series
t = np.arange(-100,101)
Y = ts(0.75,4.5,0.07,1.6,t)

#curve fitting to predict time series
def sinfunc(A,B,C,D,t):
    return A + B * np.sin(C*t + D)
(a,b,c,d), pcov = curve_fit(sinfunc, t, Y, p0=(1,5,0.06,2))
pred = sinfunc(a, b, c, d, t)

#get errors
r2 = rsq(Y,pred)

#plot curve fitting
fig = plt.figure()
plt.plot(t, pred,'r-', label = "fitted curve: MSE = %3.2f" %(mse(Y,pred)))
plt.plot(t,Y,label = "time series")
plt.xlabel("time")
plt.ylabel("y(t)")
plt.legend()
plt.title('non-linear fitting with curve fitting')
plt.savefig("1a.png")

#%%shallow learning model to predict time series 
t_train, t_test, Y_train, Y_test = train_test_split(t,Y,test_size= 0.25)
#data spliting caution becaus we have time series data, shuffeling not allowed
# t_train, t_test, Y_train, Y_test = train_test_split(t,Y,test_size= 0.25,  shuffle=False)

#%% 1 Random Forest
#building model and fitting it -> makeing predictions
rf = RandomForestRegressor(n_estimators=20)
rf.fit(t_train[:,np.newaxis],Y_train)

pred = rf.predict(t_train[:,np.newaxis])
pred_final = rf.predict(t_test[:,np.newaxis])

print("RF Training MSE: %3.3f " %(mse(Y_train,pred)))
print("RF Final MSE: %3.3f " %(mse(Y_test,pred_final)))

#plot random forest
fig1 = plt.figure()
plt.plot(t_test, pred_final,'.', label = "random forest: MSE = %3.2f" %(mse(Y_test,pred_final)))
plt.plot(t_test,Y_test,'.', label = "time series")
plt.xlabel("time")
plt.ylabel("y(t)")
plt.legend()
plt.title('non-linear fitting with random forest')
plt.savefig("1b.png")

# 2 Support Vector Machines
sv = svm.SVR()
sv.fit(t_train.reshape(-1,1),Y_train)
print("Testing score: ,", sv.score(t_test.reshape(-1,1),Y_test))

#%%deep learning
#custom activation function
def sin_act(x):
    return tf.sin(x)
#lsin = keras.layers.Dense(100,activation=sin_act)

#build model
inp = keras.layers.Input(shape=[1])
lsin = keras.layers.Dense(100,activation=sin_act) (inp)
lc = keras.layers.Concatenate()([inp,lsin])
out = keras.layers.Dense(1, activation = "linear") (lc)
model2 = keras.models.Model(inputs=[inp], outputs=[out])

#compile model
model2.compile(loss="mse", optimizer="sgd",metrics=['mae'])
history = model2.fit(t_train, Y_train, epochs= 20, validation_split=0.1)

#evaluating
print(model2.evaluate(t_test,Y_test, verbose= 0))
pd.DataFrame(history.history).plot()

#predicting 
pred = model2.predict(t_test)
print("Deep learning R2: %3.3f, MSE: %3.3f " %(rsq(Y_test, pred), mse(Y_test, pred)))

fig = plt.figure()
plt.plot(t_test, pred, '.', label = "deep learning: MSE = %3.2f" %(mse(Y_test,pred)))
plt.plot(t_test,Y_test,'.', label = "time series")
plt.xlabel("time")
plt.ylabel("y(t)")
plt.legend()
plt.title('non-linear fitting with DNN')

#%%custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return K.sin(x)
get_custom_objects().update({'custom_activation': Activation(custom_activation)})

#build model
model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=[1]))
model.add(Activation(custom_activation, name='SpecialActivation'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dense(100,activation=sin_act))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Dropout(rate=0.1))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, activation="linear")) #singel output neuron

print(model.summary())

#training
model.compile(loss="mse", optimizer="sgd",metrics=['mae'])
#es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta= 0.001, patience=10, restore_best_weights=True)
history = model.fit(t_train, Y_train, epochs= 50, validation_split=0.2)
#plotting learning curves
pd.DataFrame(history.history).plot()

#evaluating
print(model.evaluate(t_test,Y_test, verbose= 0))

#predicting 
pred = model.predict(t_test)
print("Deep learning R2: %3.3f, MSE: %3.3f " %(rsq(Y_test, pred), mse(Y_test, pred)))

fig = plt.figure()
plt.plot(t_test, pred, '.', label = "deep learning: MSE = %3.2f" %(mse(Y_test,pred)))
plt.plot(t_test,Y_test,'.', label = "time series")
plt.xlabel("time")
plt.ylabel("y(t)")
plt.legend()
plt.title('non-linear fitting with DNN')

#%% Exercises 2
#load data & examine and check quality
df = pd.read_csv("elnino-80-98.csv")
print(df)
print(df.columns)
print(df.describe().T)
fig3 = plt.figure(figsize=(20,10))
df.plot(subplots=True, layout=(4,4))
plt.title("subplots of data")
plt.savefig("2_subplot.pdf")
# SST3m and SST1y are complet
# missing values for Zonal_Winds, Meridional_Winds, Humidity, Air_Temp, SST and Nino34 but only a few. 

#coreelaton between features
print(df.corr())
f4 = plt.figure()
plt.pcolormesh(df.corr())
cbar = plt.colorbar()
plt.title("correlation between data")
#year and data are strong corelated
#SST and Air Temperature are collreated with 0.94
#SST amd SST3m are corellated with 0.74 whicht makes it probably usfull to use SST over Air Temp (I guess it will be the most important feature)

#drop features and get predictabales
#drop rows with missing SST1y - non
Y = df['SST1y'] #predictor variable

#drop features not to be used 
# year and day are not relevant, month is left to see if there is a seasonal effect, data and observation number is not usefull, latitude changes not much so assumed to stay constan and is not relevant
# SST3m probably would be useful but not available for future. SST1y shell be predicted
# Air Temp is droped becaus it is highly correlated with SST
df1 = df.drop(['Unnamed: 0','Observation','Year', 'Day', 'Date', 'Latitude', 'SST3m', 'SST1y', 'Air_Temp'], axis=1)

f5 = plt.figure()
plt.pcolormesh(df1.corr())
cbar = plt.colorbar()
plt.title("correlation between data left")


#df11 = df1.drop(['Month', 'Longitude'], axis = 1)
#df11 = pd.get_dummies(df1, columns=['Month'])
df11 = df1

#filling out missing values
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(df11) #features

#split the data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.25)

#Train a random forest model 
rf = RandomForestRegressor(n_estimators=20, max_depth=10)
rf.fit(X_train, Y_train)
print("Training score:", rf.score(X_train, Y_train))
print("Testing score:", rf.score(X_test, Y_test))
#Here, we correctly perdict the SST after one year with probability of 70 % on the training set and with 67 % on the testing set. 
# which is not perfect. 

#validation and fine-tuning of the model 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X_train, Y_train, cv=10)
print("Cross-validation-scores:", scores )
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())

#plot and pring importance of features
fig6 = plt.figure()
for i in range(np.shape(X)[1]):
    print(df11.columns[i], rf.feature_importances_[i])
plt.bar(df11.columns,rf.feature_importances_)
plt.xticks(rotation = 45)
plt.title('Importance of features')

#drope humidity and winds
X_train=pd.DataFrame(data=X_train, columns=df11.columns)
X_trainr=X_train.drop(X_train.columns[14:30],axis=1)
X_trainr.drop(['Zonal_Winds','Meridional_Winds', 'Humidity'], axis=1, inplace=True)
X_test=pd.DataFrame(data=X_test, columns=df11.columns)
X_testr=X_test.drop(X_test.columns[14:30],axis=1)
X_testr.drop(['Zonal_Winds','Meridional_Winds', 'Humidity'], axis=1, inplace=True)

#train new model
rfr = RandomForestRegressor(n_estimators=20, max_depth=10)
rfr.fit(X_trainr, Y_train)
print("Training score:", rfr.score(X_trainr, Y_train))
print("Testing score:", rfr.score(X_testr, Y_test))

#perform cross validation
plt.bar(X_testr.columns,rfr.feature_importances_)
scores = cross_val_score(rfr, X_trainr, Y_train, cv=10)
print("Cross-validation-scores:", scores )
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())

#Test the model with predicting 
pred = rfr.predict(X_trainr)
pred_final = rfr.predict(X_testr)

print("RF Training MSE: %3.3f " %(mse(Y_train,pred)))
print("RF Final MSE: %3.3f " %(mse(Y_test,pred_final)))

#Correlation of SST observed and predicted
obs = Y_test.to_numpy()
print("Corelation: %3.3f" %(np.corrcoef(pred_final, obs)[1][0]))

#scatterplot obs against predicted
obs = Y_test.to_numpy()
plt.close()
fig7 = plt.figure(figsize=(7,7))
plt.scatter(obs,pred_final, s = 0.5)
plt.ylim(20,31)
plt.xlim(20,31)
plt.title("SST after one year - Cor Coef: %3.3f" %np.corrcoef(pred_final, obs)[1][0] )
plt.ylabel("predicted")
plt.xlabel("observed")
plt.savefig("2scat.pdf")


































