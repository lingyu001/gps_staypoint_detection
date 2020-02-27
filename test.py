
# coding: utf-8

# In[14]:


import pandas as pd
import solution.ardata_cleaning as ac
import solution.feature_create as fc
from solution.train_kmeans import fit_kmeans


# In[15]:


if __name__ == "__main__":
    
    print('Test the preliminary analysis pipeline...Outputing the resmpaled data and user feature data (.csv)...')
    ## Read data
    df = pd.read_csv('df_test.csv')
    
    ## Resample data
    df_rs = ac.ardata_pipe(df,'5T')
    df_rs = df_rs[(df_rs['travelSpeed'] < 1000) & (df_rs['travelSpeed'] > 0)]
    df_rs.to_csv('df_rs_test.csv')
    
    ## Feature engineering
    X_train = fc.create_userFeature(df_rs)
    
    ## kmeans clustering
    X_train  = X_train.fillna(0)
    X_train = fit_kmeans(X_train,k=3)
    X_train.to_csv('X_train.csv')

