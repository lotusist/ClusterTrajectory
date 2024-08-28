# from functions import *
# import os, sys
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.preprocessing import MinMaxScaler


port_A = '광양항'
port_B = '포항항'
route_save_dir = './cleared_routes'

AIS_NOV = './ais_nov.pickle'
LTEM_NOV = './ltem_nov.pickle'
PORT_BOUND_JSON = '_.json'

columns_to_use = ['ROUTE_ID', 'RECPTN_DT', 'LA', 'LO', 'SOG', 'COG', 'TYPE', 'TON', 'NETWORK', 'waypoint_order', 'LA_norm', 'LO_norm', 'COG_norm', 'SOG_norm', 'visited_A', 'visited_B', 'SHIP_ID']

#get GMT Data and concat
def get_data_all():
    ais_dict = pickle.load(open(AIS_NOV,'rb'))
    ltem_dict = pickle.load(open(LTEM_NOV,'rb'))
    all_df = pd.DataFrame(columns=['SHIP_ID', 'RECPTN_DT', 'LA', 'LO', 'SOG', 'COG', 'TYPE', 'TON', 'NETWORK'])
    for ship_type, ltem_data in ltem_dict.items():
        if ship_type in ['yacht', 'fish']: continue
        ltem_data.columns = ['SHIP_ID', 'RECPTN_DT', 'LA', 'LO', 'SOG', 'COG', 'TYPE', 'TON']
        ais_data = ais_dict[ship_type].copy()
        ais_data['NETWORK'] = 'AIS'
        ltem_data['NETWORK'] ='LTEM'
        all_df = pd.concat([ltem_data,ais_data,all_df])
    all_df=all_df.sort_values('RECPTN_DT').reset_index(drop=True)

    return all_df

def preprocessData(data):
    data['RECPTN_DT'] = pd.to_datetime(data['RECPTN_DT'])
    data['date'] = data['RECPTN_DT'].dt.date
    data['date'] = data['RECPTN_DT'].dt.strftime('%Y-%m-%d')
    data['id_date'] = data['SHIP_ID'].astype(str) + '_' + data['date']

    data[['LA_norm', 'LO_norm', 'SOG_norm']] = MinMaxScaler().fit_transform(data[['LA', 'LO', 'SOG']])

    data['COG_rad'] = np.radians(data['COG']) 
    data['COG_norm'] = (np.sin(data['COG_rad']) + 1) / 2  

    return data

def get_grid(data, grid_space = 5):
    # Constants
    grid_spacing = grid_space  # 5 km separation
    earth_radius = 6371  # Earth radius in kilometers

    # Calculate the grid size in degrees
    grid_size_deg = (grid_spacing / earth_radius) * (180 / np.pi)

    # Create a grid of latitude and longitude points
    min_lat, max_lat = data['LA'].min(), data['LA'].max()
    min_lon, max_lon = data['LO'].min(), data['LO'].max()

    lat_grid = np.arange(min_lat, max_lat, grid_size_deg)
    lon_grid = np.arange(min_lon, max_lon, grid_size_deg)
    grid = [lat_grid, lon_grid]
    return grid

def get_bound_indices(port_name, grid):
    bound = list(json.load(open(PORT_BOUND_JSON))[port_name])
    [minLon, minLat, maxLon, maxLat] = bound
    lat_indices = np.searchsorted(grid[0],[minLat, maxLat], side="right") - 1
    lon_indices = np.searchsorted(grid[1], [minLon, maxLon], side="right") - 1
    bound_indices = [lat_indices, lon_indices]
    return bound_indices

def bound_filter_AB_port(data, grid, A_bound_indices, B_bound_indices):
    lat_grid, lon_grid = grid[0], grid[1]
    A_lat_indices, A_lon_indices = A_bound_indices[0], A_bound_indices[1]
    data['visited_A'] = np.logical_or.reduce([data['LA'].between(lat_grid[A_lat_indices[0]-1], lat_grid[A_lat_indices[1]+1]) & data['LO'].between(lon_grid[A_lon_indices[0]-1], lon_grid[A_lon_indices[1]+1])])
    B_lat_indices, B_lon_indices = B_bound_indices[0], B_bound_indices[1]
    data['visited_B'] = np.logical_or.reduce([data['LA'].between(lat_grid[B_lat_indices[0]-1], lat_grid[B_lat_indices[1]+1]) & data['LO'].between(lon_grid[B_lon_indices[0]-1], lon_grid[B_lon_indices[1]+1])])
    filtered = data.groupby('id_date').filter(lambda group : (group['visited_A'].any() & group['visited_B'].any()))
    
    filtered = filtered.reset_index(drop=True)
    filtered = filtered.sort_values('RECPTN_DT').reset_index(drop=True)

    ids = list(filtered['id_date'].unique())
    route_ids_idx = {}
    for id in ids:
        route_ids_idx[id] = ids.index(id)
    filtered['id_date_num'] = filtered['id_date'].map(route_ids_idx)

    return filtered

def mark_clear(aroute, mode='forward'): 

    assert ((mode=='forward') or (mode =='backward')), "type either 'forward' or 'backward' in mode."

    aroute['clear'] = False 
    visited_A_flag = False # I have visited Port A!
    visited_B_flag = False # I have visited Port B!

    if mode =='backward':
        aroute = aroute.iloc[::-1].reset_index(drop=True)
        reversed_counter = len(aroute) -1

    for index, row in aroute.iterrows():
        if mode =='backward': index = (reversed_counter - index)

        if (not visited_A_flag) and (not visited_A_flag): # FG_A = F  and  FG_B = F 
            if (row['visited_A']) or (row['visited_B']): # v_A  or  v_B = T
                visited_A_flag = row['visited_A']
                visited_B_flag = row['visited_B']

                # and start record clear waypoints
                aroute.at[index, 'clear'] = True
            else: # v_A = F or  v_B = F
                continue

        else: # FG_A or FG_B = T                
            if (row['visited_A'] != visited_A_flag) and (row['visited_B'] != visited_B_flag): # arrived in port B
                aroute.at[index, 'clear'] = True
                visited_A_flag = row['visited_A']
                visited_B_flag = row['visited_B']
                break

            else: 
                aroute.at[index, 'clear'] = True
        
    return aroute


def main():
    # takes 3 mins 30 secs
    print('1. get_data_all .... takes 3 mins 30 secs')
    all_df = get_data_all()
    print('2. preprocessData ....')
    all_df = preprocessData(all_df)
    print('3. get_grid & get_bound_indices ....')
    grid = get_grid(all_df)
    bound1 = get_bound_indices(port_A, grid)
    bound2 = get_bound_indices(port_B, grid)

    # takes 3 mins
    print('4. bound_filter_AB_port .... takes 3 mins')
    filtered_df = bound_filter_AB_port(all_df, grid, bound1, bound2)

    routes = []
    for i in list(filtered_df['id_date'].unique()):
        route = filtered_df[filtered_df['id_date'] == i].copy()
        route = route.sort_values('RECPTN_DT')
        route['waypoint_order'] = route['RECPTN_DT'].rank(method='first').astype(int)
        routes.append(route)
    
    # takes 1 min
    print('5. mark_clear points .... takes 1 min')
    clears = []
    counter = 0
    for aroute in routes:
        aroute = mark_clear(aroute, mode='forward')
        clear_aroute = aroute[aroute['clear'] == True].copy()  
        is_continuous = all(clear_aroute['waypoint_order'].diff().fillna(1) == 1)
        if (is_continuous) & (clear_aroute['visited_A'].iloc[-1]) | (clear_aroute['visited_B'].iloc[-1]): 
            clear_aroute['ROUTE_ID'] = str(counter) + '_' + clear_aroute['id_date'].astype(str)
            clears.append(clear_aroute) 
            counter += 1   
        
        aroute = mark_clear(aroute, mode='backward')
        clear_aroute = aroute[aroute['clear'] == True].copy()  
        is_continuous = all(clear_aroute['waypoint_order'].diff(-1).fillna(1) == 1)
        if is_continuous & (clear_aroute['visited_A'].iloc[-1]) | (clear_aroute['visited_B'].iloc[-1]): 
            clear_aroute = clear_aroute.iloc[::-1].reset_index(drop=True)
            clear_aroute['ROUTE_ID'] = str(counter) + '_' + clear_aroute['id_date'].astype(str)
            clears.append(clear_aroute)
            counter += 1 

    print(f'{len(clears)} clear routes have been extracted!')  

    print('6. concatenating clear routes')
    concat_clears = pd.concat(clears, ignore_index=True)
    concat_clears = concat_clears[columns_to_use]

    print(f'7. saving {port_A}_{port_B}_routes_{len(clears)}.pkl in {route_save_dir}... ')
    pickle.dump(concat_clears, open(f'{route_save_dir}/{port_A}_{port_B}_routes_{len(clears)}.pkl','wb'))

    print(f'{route_save_dir}/{port_A}_{port_B}_routes_{len(clears)}.pkl has been saved!')

if __name__ == '__main__':
    main()

