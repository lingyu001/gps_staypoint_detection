#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Kmeans clustering module 
==================
This module contains functions to 
1. select parameter k for kmeans
2. train kmeans cluster and attaching group label for each user

According to the traveler features constructed
on the Arrivalist interview data (User GPS timestamps dataset)
"""

import numpy as np
import pandas as pd
import datetime as dt


# # K-means pipeline

# In[142]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[156]:


def select_k(X_train,kmax = 15):
    """
    Input the training samples
    standardized the features to fit the k-means clusters
    Ouptut the inertia (total distance) of selecting each k 
    ----------
    Parameters
    ----------
    kmax = the maximum candidate k value 
    
    """
    
    scaler = StandardScaler()
    scaler.fit(user_features_travel)
    
    X = scaler.transform(X_train)
    distortions = []
    for i in range(1, kmax):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)
    
    return distortions


# In[154]:


def fit_kmeans(X_train, k = 6, tol = 1e-04):
    """
    Input the training samples
    standardized the features and fit the k-means clusters
    Ouptut the data labeled with the selected k clustering group
    ----------
    Parameters
    ----------
    k = the number of clusters as well as number of centroids to generate.
    tol = tolerance with regards to inertia to declare convergence
    
    """
    
    scaler = StandardScaler()
    
    kmeans = KMeans(
        n_clusters=k, init='random',
        n_init=10, max_iter=300,
        tol=tol, random_state=0
    )
    
    kmeans_pipe = Pipeline([
    ('StandardizeScaler',scaler), 
    ('Kmeans',kmeans)
    ])
    cluster_result = kmeans_pipe.fit(X_train)
    X_train['km_cluster'] = cluster_result.steps[1][1].labels_
    
    return X_train


# In[ ]:


if __name__ == "__main__":
    
    print('Kmeans clustering module')

