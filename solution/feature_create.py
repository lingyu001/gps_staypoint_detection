#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Feature engineering module 
==================
This module contains functions to 
1. detect stay points of each user, create stay point features
2. Create App use frequency features for each user
3. Create travel type features for each user

on the Arrivalist interview data (User GPS timestamps dataset)
"""

import numpy as np
import pandas as pd
import datetime as dt


# In[2]:


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


# # Feature Engineering

# In[3]:


def detect_stayPoints(df_rs, stopDuration = 1):
     
    """
    Detect stay points of each user using spatial clustering,
    Input the resampled users GPS data as the pandas dataframe,
    Output the detected stay points in each day 
    ----------
    Parameters
    ----------
    df : pandas dataframe
        contains GSP samples for all users
    stopDuration: a time interval parameter to define the stay point
        taking integer value
        One unit indicating a 5 mins time duration
        eg. stopDuration = 1 indicates a location that user spent > 5 mins in a day could be defined as a stay point.
            stopDuration = 2 indicates a 10 mins stop in a day, etc.
    ----------
    Stay point detection
    ----------
    1. A stay point is defined as a geo-location with a 100 meter radius circle.
    2. A user's GPS data sample appear in 100 meter circle more than a defined stay time duration in a day consider a stay point.
    3. The stay time duration is defined by the stopDuration parameter, default = 1 : stop time > 5 mins indicate a stay point.
    
    """

    ## User-Stay point DataFrame, keep it for future analysis
    df_stayPoints = df_rs[['device_id','ts_date','lat_lon_3','ts']]                                                 .groupby(['device_id','ts_date','lat_lon_3']).count().reset_index()
    df_stayPoints = df_stayPoints.drop(df_stayPoints[df_stayPoints['ts']<=stopDuration].index).reset_index()                                                 .rename(columns={'lat_lon_3':'stayPoint','ts':'stopFreq'})

    df_stayPoints['MonFri'] = df_stayPoints['ts_date'].map(lambda x: (x.weekday() < 5)*1)
    
    return df_stayPoints


# In[4]:


def user_stopnum(df_stay):
     
    """
    Construct user stay point feature: average number of user's stay point in weekdays and weekends,
    Input the user stay points in all days as the pandas dataframe,
    Output the average detected stay points of weekday and weekend for each user
 
    ----------
    Average Stay points
    ----------
    
    Average Stay points = (number of stay point detected in all weekdays/weekends) / (number of weekdays/weekends)
    
    """

    df_stayPointsnum = df_stay[['device_id','ts_date','MonFri','stayPoint']].groupby(['device_id','ts_date'])                                                                 .agg({'stayPoint':'count','MonFri':'first'})                                                                .rename(columns={'stayPoint':'staypointNum'}).reset_index()

    ## User average weekday and weekend Stay point Dataframe
    df_userStay = df_stayPointsnum[df_stayPointsnum['MonFri']==1][['device_id','staypointNum']]                                     .groupby('device_id').mean().rename(columns={'staypointNum':'weekdayStay'})                                        .join(df_stayPointsnum[df_stayPointsnum['MonFri']==0][['device_id','staypointNum']]                                         .groupby('device_id').mean().rename(columns={'staypointNum':'weekendStay'})).fillna(0)
            
    return df_userStay.fillna(0)


# In[5]:


def detect_useFreq(df_rs):
     
    """
    Construct app usage frequency features
    
    Input the resampled data for all users as the pandas dataframe,
    Output a dataframe including two app usage frequency features for each user 
    
    1. app use frequency during use days = all the data points acquired / the days in a year received data 
    2. Daily app use frequency = all the data points acquired / (last day - first day in a year received data)
 

    """
    user_freq = pd.DataFrame(df_rs[['device_id','ts']].groupby('device_id')['ts']                              .apply(lambda x: x.count() / ((x.max() - x.min()).days + 1)))                              .rename(columns = {'ts':'avgDailyFreq'})

    freq_day = df_rs[['device_id','ts_date','ts']].groupby(['device_id','ts_date']).count().groupby('device_id').mean()                                                      .rename(columns={'ts':'avgUsedayFreq'})
    user_freq = user_freq.join(freq_day)
    
    return user_freq


# In[6]:


def detect_travelerType(df, region, period):
    """
    Detect traveler type regarding the average number of presence in different city, State, and Country across a time period
    Input the users GPS data as the pandas dataframe,
    Output the average number of city, State,and country traveled during a period
    ----------
    Parameters
    ----------
    df : pandas dataframe
        contains GSP samples for all users
    region: {'country','region','city'}
            region in U.S. indicate State
    
    period: {'year','month','week'}
    
    """
    avg_reg_pd = df.groupby(['device_id',period])[region].nunique()                 .reset_index()[['device_id',region]].groupby('device_id').mean()                 .rename(columns = {region:'avg{}{}'.format(region,period)})
    
    return avg_reg_pd


# In[7]:


def aggre_travelType(df_rs):
    """
    Input the resampled GPS data for all users as a pandas dataframe
    Ouptut the all the travel frequency freatures for each user as a pandas dataframe
    
    """
    travel_agg = []
    for region in ['country','region','city']:
        for period in ['year','month','week']:
            travel_agg.append(detect_travelerType(df_rs,region,period))
    travel_type = pd.concat(travel_agg,axis=1)
    
    return travel_type


# In[8]:


def create_userFeature(df_rs):

    user_features_travel = pd.concat([detect_useFreq(df_rs),user_stopnum(detect_stayPoints(df_rs)),aggre_travelType(df_rs)],axis=1)
    
    return user_features_travel


# In[9]:


if __name__ == "__main__":
    
    print('Traveler Feature Engineering module' )
    
    import ardata_cleaning as ac
    ## Full sample
    #df = get_ar_data()
    
    ## Test sample
    df = pd.read_csv('df_test.csv')
    
    df_rs = ac.ardata_pipe(df,'5T')
    
    print('producing the traveler features for each user as a dataframe...full sample takes a while...')
    X_train = create_userFeature(df_rs)
    X_train.to_csv('X_train.csv')
    
    print(X_train.info())
    print("Output user features to 'X_train.csv'")

