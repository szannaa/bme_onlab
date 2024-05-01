#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[26]:


import pandas as pd


# In[27]:


import xml.etree.ElementTree as et


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


import os


# In[30]:


import sys
import shutil


# # Functions

# In[31]:


def transform_xml(xml_doc):
    attr = xml_doc.attrib
    for xml in xml_doc.iter('vehicle'):
        _dict = attr.copy()
        _dict.update(xml.attrib)
        
        yield _dict


# In[32]:


price_per_liter = 624 / 1.27
eta = 0.45
m = 18000
g = 9.81
mg_to_liter = 0.0000011765
liter_to_joule = 38000000


# In[33]:


def get_price_up(emission, delta_h):
    emission = emission * mg_to_liter * liter_to_joule
    emission = emission + (1/eta) * m * g * delta_h
    emission = emission * (1/liter_to_joule)
    emission = emission * price_per_liter
    return emission


# In[34]:


def get_price_down(emission, delta_h):
    emission = emission * mg_to_liter * liter_to_joule
    emission = emission - (1/eta) * m * g * delta_h
    emission = emission * (1/liter_to_joule)
    emission = emission * price_per_liter
    return emission


# In[35]:


def get_last_folder(path):
    # Normalize the path to handle different separators and remove trailing separator
    normalized_path = os.path.normpath(path)
    # Split the path into components
    folders = normalized_path.split(os.sep)
    # Get the last folder
    last_folder = folders[-1]
    return last_folder


# In[36]:


def calc_elevation_up(group):
    z_diff = pd.to_numeric(group['z']).diff()

    # Filter out negative differences (upward movement)
    up = z_diff.apply(lambda x: x if x > 0 else 0)

    # Sum the positive differences to get the total upward movement
    total_up = up.sum()
    return total_up


# In[37]:


def calc_elevation_down(group):
    z_diff = pd.to_numeric(group['z']).diff()

    # Filter out negative differences (upward movement)
    down = z_diff.apply(lambda x: x if x < 0 else 0)

    # Sum the positive differences to get the total upward movement
    total_down = down.sum()
    return total_down


# In[38]:


def transform_xml_tripinfo(xml_doc):
    attr = xml_doc.attrib
    for xml in xml_doc.iter('tripinfo'):
        _dict = attr.copy()
        _dict.update(xml.attrib)
        
        yield _dict


# In[39]:


def get_routeids_df(base_folder):
    file_path = os.path.join(base_folder, "trips.txt")
    return pd.read_csv(file_path, sep=",")


# In[40]:


def get_route_id(trip_id, trip_ids):
    trip_id = trip_id[:-2]
    line = trip_ids.loc[trip_ids['trip_id'] == trip_id]
    route_id = line.iloc[0]['route_id']

    return route_id


# In[41]:


def transform_xml_stops(xml_doc):
    for route in xml_doc.iter('route'):
        route_dict = route.attrib.copy()
        stops = []
        
        # Iterate over each <stop> element within the current <route> element
        for stop in route.findall('stop'):
            stop_dict = stop.attrib.copy()
            stops.append(stop_dict)
        
        # Include stops in the route dictionary
        route_dict['stops'] = stops
        
        yield route_dict


# In[ ]:


def main(seed, scale, simulation_folder):
    
    seed = sys.argv[1]
    scale = sys.argv[2]
    simulation_folder = sys.argv[3]
    print(seed)
    print(scale)
    print(simulation_folder)


    # # XML to df

    # In[42]:


    base_folder = "C:\\Users\\Admin\\Sumo\\" + simulation_folder

    print(base_folder)

    # In[43]:


    file_path = os.path.join(base_folder, "emission.out.xml")
    shutil.copyfile(file_path, f"output\\{simulation_folder}_emission_{seed}_{scale}")
    emission_output = et.parse(file_path)

    transform = transform_xml(emission_output.getroot())
    emission_output_list = list(transform)

    emission_output_df = pd.DataFrame(emission_output_list)
    emission_output_df = emission_output_df.drop(emission_output_df.columns[0], axis=1)

    #emission_output_df.shape


    # In[44]:


    #emission_output_df


    # In[45]:


    file_path = os.path.join(base_folder, "tripinfo.xml")
    b_tripinfo_output = et.parse(file_path)

    transform = transform_xml_tripinfo(b_tripinfo_output.getroot())
    b_tripinfo_output_list = list(transform)

    b_tripinfo_output_pd = pd.DataFrame(b_tripinfo_output_list)
    b_tripinfo_output_pd = b_tripinfo_output_pd.drop(b_tripinfo_output_pd.columns[0], axis=1)
    #b_tripinfo_output_pd


    # In[46]:


    file_path = os.path.join(base_folder, "gtfs_pt_vehicles.add.xml")
    stops = et.parse(file_path)

    transform = transform_xml_stops(stops.getroot())
    stops_list = list(transform)

    stops_pd = pd.DataFrame(stops_list)


    # In[47]:


    file_path = os.path.join(base_folder, "gtfs_pt_vehicles.add.xml")
    vehicles = et.parse(file_path)

    transform = transform_xml(vehicles.getroot())
    vehicles_list = list(transform)

    vehicles_pd = pd.DataFrame(vehicles_list)
    vehicles_pd = vehicles_pd.drop(vehicles_pd.columns[0], axis=1)
    #vehicles_pd


    # In[48]:


    route_ids_df = get_routeids_df(base_folder)


    # # Group by id

    # In[49]:


    grouped_df = emission_output_df.groupby('id')

    list_of_dfs = [group_data for _, group_data in grouped_df]


    # In[50]:


    results = []
    for group_id, group_data in grouped_df:
        avg_speed = group_data['speed'].astype(float).mean()

        time_loss = b_tripinfo_output_pd.loc[b_tripinfo_output_pd['id'] == group_id, 'timeLoss'].values[0]
        route_length = b_tripinfo_output_pd.loc[b_tripinfo_output_pd['id'] == group_id, 'routeLength'].values[0]
        
        route = vehicles_pd.loc[vehicles_pd['id'] == group_id, 'route'].values[0]
        count_stops = stops_pd[stops_pd['id'] == route]['stops'].apply(len).sum()
        
        z_up = calc_elevation_up(group_data)
        z_down = calc_elevation_down(group_data)
        
        d_height = z_up + z_down
        
        fuel_sum = group_data['fuel'].astype(float).sum()
        
        route_id = get_route_id(group_id, route_ids_df)
        
        if(d_height >= 0):
            fuel_sum = get_price_up(group_data['fuel'].astype(float).sum(), d_height)
        else:
            fuel_sum = get_price_down(group_data['fuel'].astype(float).sum(), d_height)
        
        # Store the results in a dictionary
        group_result = {
            'routeid': route_id,
            'id': group_id,
            'avgSpeed': avg_speed,
            'fuel': fuel_sum,
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

    # In[51]:


    tableEmission = pd.read_csv('emissionData.csv', delimiter=';')

    #tableEmission.shape


    # ## settings

    # In[52]:


    locSetting = get_last_folder(base_folder)
    seedSetting = seed
    trafficScaleSetting = scale


    # In[53]:
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
            'emission': row['fuel']
        }
        new_rows.append(row_data)
        
    new_rows_df = pd.DataFrame(new_rows)
    tableEmission = tableEmission.dropna(axis=1, how='all')

    tableEmission = pd.concat([tableEmission, new_rows_df], ignore_index=True)


    # In[54]:


    tableEmission.to_csv('emissionData.csv', index=False, sep=';')
    print(tableEmission)


    # In[ ]:


if __name__ == "__main__":
    main(seed = 0, scale = 0, simulation_folder = "")

