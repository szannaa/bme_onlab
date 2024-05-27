#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy.stats import shapiro


# In[2]:


def add_columns_per_m(df):
    df['stops_per_m'] = df['numOfStops'] / df['route_length']
    df['emission_per_m'] = df['emission'] / df['route_length']
    df['elevation_up_per_m'] = df['elevation_up'] / df['route_length']
    df['elevation_down_per_m'] = df['elevation_down'] / df['route_length']
    df['timeloss_per_m'] = df['timeloss'] / df['route_length']

    return df


# In[3]:


def drop_unnecessary_columns(df):
    return df.drop(['routeId', 'tripId', 'loc', 'seed'], axis=1)


# In[4]:


def avg_speed(df):
    max_value = df['avgSpeed'].max()
    df['avgSpeed'] = df['avgSpeed'] / max_value
    return df


# In[5]:


def drop_stops_and_route_length(df):
    return df.drop(['numOfStops', 'route_length'], axis=1)


# In[6]:


def drop_redundant_columns(df):
    return df.drop(['numOfStops', 'route_length', 'emission', 'elevation_up', 'elevation_down', 'timeloss'], axis=1)


# In[7]:


def normalize(df):
    return df.apply(lambda x: x / abs(x).max(), axis=0)


# In[8]:


def add_delta_elevation(df):
    df['elevation_down_per_m'] = df['elevation_down_per_m'].abs()
    df['delta_elevation_per_m'] = df['elevation_up_per_m'] - df['elevation_down_per_m']
    new_df = df.drop(['elevation_up_per_m', 'elevation_down_per_m'], axis=1)
    return new_df


# In[9]:


def drop_max_timeloss(df):
    max_timeloss_index = df['timeloss_per_m'].idxmax()
    df = df.drop(max_timeloss_index)
    
    return df


# In[10]:


tableEmission = pd.read_csv('C:\\Users\\Admin\\Desktop\\onlab\\bme_onlab\\data\\emissionData.csv', delimiter=';')
tableBattery = pd.read_csv('C:\\Users\\Admin\\Desktop\\onlab\\bme_onlab\\data\\batteryData.csv', delimiter=';')


# ## Drop unnecessary columns

# In[11]:


tableEmission = tableEmission[tableEmission['seed'] <= 3]
tableBattery = tableBattery[tableBattery['seed'] <= 3]


# In[12]:


tableEmission = drop_unnecessary_columns(tableEmission)
tableEmission


# In[13]:


tableBattery = drop_unnecessary_columns(tableBattery)
tableBattery


# ## create new columns from existing ones

# In[14]:


tableBattery = avg_speed(tableBattery)
tableEmission = avg_speed(tableEmission)


# In[15]:


tableBattery = add_columns_per_m(tableBattery)
tableEmission = add_columns_per_m(tableEmission)


# In[16]:


tableBattery = drop_redundant_columns(tableBattery)
tableEmission = drop_redundant_columns(tableEmission)
tableBattery

tableEmission = drop_max_timeloss(tableEmission)
tableEmission = drop_max_timeloss(tableEmission)
tableEmission = drop_max_timeloss(tableEmission)


# In[17]:


tableEmissionDeltaEle = add_delta_elevation(tableEmission)
tableBatteryDeltaEle = add_delta_elevation(tableBattery)

tableBattery = tableBattery.drop(['delta_elevation_per_m'], axis=1)
tableEmission = tableEmission.drop(['delta_elevation_per_m'], axis=1)


# In[18]:


tableBattery = normalize(tableBattery)
tableEmission = normalize(tableEmission)
#tableBatteryDeltaEle = normalize(tableBatteryDeltaEle)
#tableEmissionDeltaEle = normalize(tableEmissionDeltaEle)

tableBattery


# In[19]:


tableBattery.to_csv('battery.csv', index=False, sep=';')
tableEmission.to_csv('emission.csv', index=False, sep=';')
tableBatteryDeltaEle.to_csv('batteryDeltaEle.csv', index=False, sep=';')
tableEmissionDeltaEle.to_csv('emissionDeltaEle.csv', index=False, sep=';')


# In[20]:


for column in tableEmission.columns:
    stat, p = shapiro(tableEmission[column])
    print(f'Column: {column}, p-value: {p}')
    if p > 0.05:
        print(f'Column "{column}" appears to be normally distributed.')
    else:
        print(f'Column "{column}" does not appear to be normally distributed.')

