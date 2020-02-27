#!/usr/bin/env python
# coding: utf-8

# In[27]:


"""
GPS data resampling and cleaning module nodes
==================
This module contains functions to 
1. get data
2. data resmpling using sliding window
3. data cleaning 

on the Arrivalist interview data (User GPS timestamps dataset)
"""

import numpy as np
import pandas as pd
import requests
import datetime as dt
from geopy.distance import distance


# In[28]:


def get_ar_data(url="https://s3.amazonaws.com/arrivalist-puzzles/interview_data.csv.zip"):
    """Loads the Arrivalist data set into memory.
    Returns
    -------
    X : array-like, shape=[n_samples, n_features]
        Training data for the MNIST data set.
        
    y : array-like, shape=[n_samples,]
        Labels for the MNIST data set.
    """
    
    df = pd.read_csv(url)

    return df


# In[29]:


def convert_time(df):
    """
    Convert the ts column from string to UTCtimestamp in the dataframe
    
    """
    
    df['ts'] = pd.to_datetime(df['ts'],utc=True)
    
    return df


# In[30]:


def data_resample(df, window = "5T"):
    
    """
    Resample the timestamp data using a sliding window,
    Input the GPS data of each user,
    Output the resampled GPS data taking every first sample within the window time interval
    ----------
    Parameters
    ----------
    df : pandas dataframe
        contains device_id, ts = timestamp (str)
    window : sliding window {minutes: "T", days:"D", seconds:"S"}, default = 5T (5 mins)
        parameter refer to https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html.
    """

    df_rs = df.set_index('ts').groupby('device_id').resample('5T').first().dropna(how='any', subset=['ts_date'])                                                          .drop(['device_id'],axis=1).reset_index()
    return df_rs


# In[31]:


def pair_latlon(df):
    """
    create a column of (latitude, longitude) in the dataframe,
    rounded each (lat,long) value to 0.001 (for clustering)
    *One 0.001 decimal change in lat: ~ 111 meters

    """
    df['lat_lon_3'] = df[['lat','lon']].apply(lambda x: "{},{}".format(round(x.iloc[0],3),round(x.iloc[1],3)),axis=1)

    return df


# In[32]:


def tspeed_est(userdata):
    
    """
    Estimate the travel speed between two adjecant gps points,
    Input the GPS data of a single user,
    Output the travel speed data for the user
    Including three columns: travelTime, distance, travelSpeed
    
    ----------
    Parameter
    ----------
    df : pandas dataframe
         contains the sample for a single user
         
    ----------
    Travel speed caculation 
    ----------
    travelTime = timestamp at current GPS location - timestamp at current GPS location (hour)  
    distance = distance between current GPS location and last GPS location (km)  
    travelSpeed = distance / travelTime (km per hour)
  
         
    """
    
    row_iter = userdata.iterrows()
    last = next(row_iter)[1]
    travel_speed = pd.DataFrame(index=userdata.index,columns=['travelTime','distance','travelSpeed'])

    for i, row in row_iter:
        travel_speed.loc[i]['travelTime'] = round(((row['ts']-last['ts']).total_seconds())/3600,4)
        travel_speed.loc[i]['distance'] = round(distance(row['lat_lon_3'],last['lat_lon_3']).km,4)
        
        try:
            travel_speed.loc[i]['travelSpeed'] = round(travel_speed.loc[i]['distance'] / travel_speed.loc[i]['travelTime'],4)
        except:
            travel_speed.loc[i]['travelSpeed'] = -999
            
        last = row    
    travel_speed = travel_speed.fillna(0)
        
    return travel_speed


# In[33]:


def append_travelSpeed(df):
    """
    Append the travel speed for each User
    
    """
    gp = df.groupby('device_id')
    temp = []
    for key, data in gp:
        temp.append(tspeed_est(data))
    ts = pd.concat(temp)
    df = df.join(ts)

    return df


# In[34]:


def add_dateColumns(df):
    """
    Add year, month, week, day, weekday columns to the dataframe
    Convert ts_date from string to datetime.date
    """
    df['ts_date'] = df['ts'].map(lambda x: x.date())

    df['year'] = df['ts'].map(lambda x: x.year)
    df['month'] = df['ts'].map(lambda x: x.month)
    df['day'] = df['ts'].map(lambda x: x.day)
    df['weekday'] = df['ts'].map(lambda x: x.weekday())
    df['week'] = df['ts'].map(lambda x: x.week)
    
    return df


# In[63]:


def ardata_pipe(df, resampleWindow = '5T'):
    """
    Data resampling pipeline:
    1. Resample data using sliding window
    2. Pari latitude and longitude to one column
    3. Add travel time, travel distance and travel speed column for data cleaning
    4. Add year, month, week, day column
    ----------
    Parameters
    ----------
    df : original data as pandas dataframe
        must contains device_id, ts = timestamp (str), latitude, longitude
        
    resampleWindow : sliding window {minutes: "T", days:"D", seconds:"S"}, default = 5T (5 mins)
        parameter refer to https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html.
        
    """
    
    df_rs = add_dateColumns(append_travelSpeed(pair_latlon(data_resample(convert_time(df), window = resampleWindow))))
    
    return df_rs


# In[ ]:


if __name__ == "__main__":
    
    print('Data cleaning and resampling module, resampling data...the full sample takes a while...')
    ## Full sample
    #df = get_ar_data()
    
    ##Try a test sample
    df = pd.read_csv("df_test.csv")
    
    df_rs = ardata_pipe(df,'5T')
    df_rs.to_csv('df_rs.csv')
    print(df_rs.info())
    print('Output resampled data to "df_rs.csv"')

