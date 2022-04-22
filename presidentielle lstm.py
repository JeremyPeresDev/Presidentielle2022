#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jérémy Peres
@version:'1.0'
@website:jeremyperes.fr
@email:contact@jeremyperes.fr
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.graph_objects as go

#data from csv
df = pd.read_csv('data2.csv', encoding = 'unicode_escape', engine ='python')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
train_dates = pd.to_datetime(df['Date'], format='%d/%m/%Y')

forecast_dates = []
n_lookback = 60
n_forecast = 30

#list of predicition dates
predict_period_dates = pd.date_range(list(train_dates)[len(train_dates)-1], periods=n_forecast+1, freq='D').tolist()
predict_period_dates.pop(0)
for time_i in predict_period_dates :
    forecast_dates.append(time_i.date())
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates)})

#find model and apply for each candidates
for candidate in range(1, 13) :
    cols = list(df)[candidate+1:candidate+2]
    
    df_for_training = df[cols].astype(float)
    df_for_training = df_for_training.values.reshape(-1, 1)
    
    # scaling the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))#StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
        
    iepochs = 10
    
    trainX = []
    trainY = []
    
    #prep training data
    for i in range(n_lookback, len(df_for_training_scaled) - n_forecast +1):
        trainX.append(df_for_training_scaled[i - n_lookback: i])
        trainY.append(df_for_training_scaled[i: i + n_forecast])
    
    trainX, trainY = np.array(trainX), np.array(trainY)
       
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_lookback, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(n_forecast))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    
    # fit the model
    model.fit(trainX, trainY, epochs=iepochs, batch_size=16, verbose=1)
        
    #Make prediction
    prediction = model.predict(trainX[-n_forecast:]) 
    
    # generate the forecasts
    X_ = df_for_training_scaled[- n_lookback:] 
    X_ = X_.reshape(1, n_lookback, 1)
    
    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)
    
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
    df_forecast[cols[0]] = Y_.flatten()
    
#----- forecast only
df_forecast.to_csv("forecast.csv")

fig = go.Figure()
for i in range(1, len(df_forecast.columns)):
    fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast[df_forecast.columns[i]], name=df_forecast.columns[i]))

fig.write_html("forecast.html")

#----- data + forecast
totaldf = pd.concat([df, df_forecast], sort=False)
totaldf = totaldf.drop(columns=["Total"])
totaldf.to_csv("total and forecast.csv")

figtotal = go.Figure()

for i in range(1, len(totaldf.columns)):
    figtotal.add_trace(go.Scatter(x=totaldf['Date'], y=totaldf[totaldf.columns[i]], name=totaldf.columns[i]))

figtotal.write_html("totalforecast.html")
