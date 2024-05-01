#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import pandas as pd


# In[2]:


import xml.etree.ElementTree as et


# In[3]:


import os
import sys
import shutil


# In[4]:


import matplotlib.pyplot as plt



# # Functions

# In[6]:


def get_last_folder(path):
    # Normalize the path to handle different separators and remove trailing separator
    normalized_path = os.path.normpath(path)
    # Split the path into components
    folders = normalized_path.split(os.sep)
    # Get the last folder
    last_folder = folders[-1]
    return last_folder


# In[7]:


price_per_wh = 110 / 1000


# In[8]:


def get_price(d_energy):
    return d_energy * price_per_wh


# In[9]:


def transform_xml(xml_doc):
    attr = xml_doc.attrib
    for xml in xml_doc.iter('vehicle'):
        _dict = attr.copy()
        _dict.update(xml.attrib)
        
        yield _dict


# In[10]:


def get_route_id(trip_id, trip_ids):
    trip_id = trip_id[:-2]
    line = trip_ids.loc[trip_ids['trip_id'] == trip_id]
    route_id = line.iloc[0]['route_id']

    return route_id


# In[11]:


def get_routeids_df(base_folder):
    file_path = os.path.join(base_folder, "trips.txt")
    return pd.read_csv(file_path, sep=",")


# In[12]:


def calc_elevation_up(group):
    z_diff = pd.to_numeric(group['z']).diff()

    # Filter out negative differences (upward movement)
    up = z_diff.apply(lambda x: x if x > 0 else 0)

    # Sum the positive differences to get the total upward movement
    total_up = up.sum()
    return total_up


# In[13]:


def calc_elevation_down(group):
    z_diff = pd.to_numeric(group['z']).diff()

    # Filter out negative differences (upward movement)
    down = z_diff.apply(lambda x: x if x < 0 else 0)

    # Sum the positive differences to get the total upward movement
    total_down = down.sum()
    return total_down


# In[14]:


def transform_xml_tripinfo(xml_doc):
    attr = xml_doc.attrib
    for xml in xml_doc.iter('tripinfo'):
        _dict = attr.copy()
        _dict.update(xml.attrib)
        
        yield _dict


# In[15]:


def transform_xml_stops(xml_doc):
    for route in xml_doc.iter('route'):
        route_dict = route.attrib.copy()
        stops = []
        
        for stop in route.findall('stop'):
            stop_dict = stop.attrib.copy()
            stops.append(stop_dict)
        
        route_dict['stops'] = stops
        
        yield route_dict


# In[16]:


def get_group_by_id(list_of_dfs, desired_id):
    for df in list_of_dfs:
        if desired_id in df['id'].values:
            return df[df['id'] == desired_id]
    raise ValueError(f"ID '{desired_id}' not found in any dataframe.")


# In[ ]:


def main(seed, scale, simulation_folder):
    
    seed = sys.argv[1]
    scale = sys.argv[2]
    simulation_folder = sys.argv[3]
    print(seed)
    print(scale)
    print(simulation_folder)
    
    # # XML to df

    # In[17]:
    base_folder = "C:\\Users\\Admin\\Sumo\\" + simulation_folder


    # In[18]:


    file_path = os.path.join(base_folder, "emission.out.xml")
    shutil.copyfile(file_path, f"output\\b_emission_{seed}_{scale}")
    emission_output = et.parse(file_path)

    transform = transform_xml(emission_output.getroot())
    emission_output_list = list(transform)

    emission_output_df = pd.DataFrame(emission_output_list)
    emission_output_df = emission_output_df.drop(emission_output_df.columns[0], axis=1)

    #emission_output_df.shape


    # In[19]:


    file_path = os.path.join(base_folder, "Battery.out.xml")
    shutil.copyfile(file_path, f"output\\b_battery_{seed}_{scale}")
    
    battery_output = et.parse(file_path)
    battery_output_root = battery_output.getroot()

    transform = transform_xml(battery_output_root)
    battery_output_list = list(transform)

    battery_output_pd = pd.DataFrame(battery_output_list)

    battery_output_pd = battery_output_pd.drop(battery_output_pd.columns[0], axis=1)
    #battery_output_pd


    # In[20]:


    file_path = os.path.join(base_folder, "tripinfo.xml")

    b_tripinfo_output = et.parse(file_path)
    b_tripinfo_output_root = b_tripinfo_output.getroot()

    transform = transform_xml_tripinfo(b_tripinfo_output_root)
    b_tripinfo_output_list = list(transform)

    b_tripinfo_output_pd = pd.DataFrame(b_tripinfo_output_list)

    b_tripinfo_output_pd = b_tripinfo_output_pd.drop(b_tripinfo_output_pd.columns[0], axis=1)
    #b_tripinfo_output_pd


    # In[21]:


    file_path = os.path.join(base_folder, "gtfs_pt_vehicles.add.xml")
    stops = et.parse(file_path)

    transform = transform_xml_stops(stops.getroot())
    stops_list = list(transform)

    stops_pd = pd.DataFrame(stops_list)


    # In[22]:


    file_path = os.path.join(base_folder, "gtfs_pt_vehicles.add.xml")
    vehicles = et.parse(file_path)

    transform = transform_xml(vehicles.getroot())
    vehicles_list = list(transform)

    vehicles_pd = pd.DataFrame(vehicles_list)
    vehicles_pd = vehicles_pd.drop(vehicles_pd.columns[0], axis=1)
    #vehicles_pd


    # In[23]:


    route_ids_df = get_routeids_df(base_folder)


    # ## Grouping by id

    # In[24]:


    grouped_df = battery_output_pd.groupby('id')

    list_of_dfs = [group_data for _, group_data in grouped_df]
    #list_of_dfs


    # In[25]:


    grouped_emission_df = emission_output_df.groupby('id')

    list_of_emission_dfs = [group_data for _, group_data in grouped_emission_df]
    #list_of_emission_dfs



    # In[26]:


    for group_id, group_data in grouped_df:
        avg_speed = group_data['speed'].astype(float).mean()
        
        energy = float(group_data['totalEnergyConsumed'].iloc[-1])-float(group_data['totalEnergyRegenerated'].iloc[-1])
        #print(get_price(energy))


    # In[27]:


    results = []
    for group_id, group_data in grouped_df:
        avg_speed = group_data['speed'].astype(float).mean()
        
        energy = get_price(float(group_data['totalEnergyConsumed'].iloc[-1])-float(group_data['totalEnergyRegenerated'].iloc[-1]))
        #print(float(group_data['totalEnergyConsumed'].iloc[-1])-float(group_data['totalEnergyRegenerated'].iloc[-1]))
        
        time_loss = b_tripinfo_output_pd.loc[b_tripinfo_output_pd['id'] == group_id, 'timeLoss'].values[0]
        route_length = b_tripinfo_output_pd.loc[b_tripinfo_output_pd['id'] == group_id, 'routeLength'].values[0]
        
        route = vehicles_pd.loc[vehicles_pd['id'] == group_id, 'route'].values[0]
        count_stops = stops_pd[stops_pd['id'] == route]['stops'].apply(len).sum()
        
        route_id = get_route_id(group_id, route_ids_df)
        
        z_up = calc_elevation_up(get_group_by_id(list_of_emission_dfs, group_id))
        z_down = calc_elevation_down(get_group_by_id(list_of_emission_dfs, group_id))
        
        # Store the results in a dictionary
        group_result = {
            'routeid': route_id,
            'id': group_id,
            'avgSpeed': avg_speed,
            'battery': energy,
            'timeloss': time_loss,
            'routeLength': route_length,
            'numOfStops': count_stops,
            'up': z_up,
            'down': z_down
        }
        
        # Append the dictionary to the results list
        results.append(group_result)

    # Convert the results list to a DataFrame
    result_df = pd.DataFrame(results)
    #print(result_df)


    # # CSV

    # In[28]:


    tableBattery = pd.read_csv('batteryData.csv', delimiter=';')

    #tableBattery.shape


    # ## settings

    # In[29]:


    locSetting = get_last_folder(base_folder)
    seedSetting = seed
    trafficScaleSetting = scale


    # ## Df to csv

    # In[30]:
    new_rows = []
	
    for index, row in result_df.iterrows():
        row_data = {
            'routeId': row['routeid'],
            'loc': locSetting,
            'tripId': row['id'],
            'seed': seedSetting,
            'avgSpeed': row['avgSpeed'],
            'timeloss': row['timeloss'],
            'route_length': row['routeLength'],
            'elevation_up': row['up'],
            'elevation_down': row['down'],
            'trafficScale': trafficScaleSetting,
            'numOfStops': row['numOfStops'],
            'emission': row['battery']
        }
        new_rows.append(row_data)
        
        #any empty or all-NA columns in tableBattery are excluded before concatenating the DataFrames
        
    new_rows_df = pd.DataFrame(new_rows)
    tableBattery = tableBattery.dropna(axis=1, how='all')

    tableBattery = pd.concat([tableBattery, new_rows_df], ignore_index=True)


    # In[31]:


    tableBattery.to_csv('batteryData.csv', index=False, sep=';')
    print(tableBattery)


# In[ ]:


if __name__ == "__main__":
    main(seed = 0, scale = 0, simulation_folder = "")

