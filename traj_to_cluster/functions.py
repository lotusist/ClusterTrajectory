from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os
import time
from tslearn.metrics import dtw 
import json


PORT_BOUND_JSON = '_.json'

def preprocessData(data):
    print("Preprocessing Data...")
    time_start = time.time()
    #data type processing
    data = data.astype({'szMsgSendDT':'int'})
    data['szMsgSendDT'] = data['szMsgSendDT']/1000
    data['szMsgSendDT'] = pd.to_datetime(data['szMsgSendDT'], format = "%Y%m%d%H%M%S")
    data = data[(data['dLat'] > 31) & (data['dLat'] < 39) & (data['dLon'] > 124) & (data['dLon'] < 132)]

    data['dLat_tr'] =  MinMaxScaler().fit_transform(data[['dLat']])
    data['dLon_tr'] =  MinMaxScaler().fit_transform(data[['dLon']])
    data['dCOG_tr'] =  MinMaxScaler().fit_transform(data[['dCOG']])
    data['dSOG_tr'] =  MinMaxScaler().fit_transform(data[['dSOG']])
    data.index = data['szMsgSendDT']
    time_end = time.time()
    print("Done with ", time_end-time_start)

    return data


def collectData(path):
   
    # # Data concat
    print("Collecting data...")
    time_start = time.time()
    # setting the path for joining multiple files
    path = os.path.join(path, "*.csv")

    # list of merged files returned
    merged = glob.glob(path)


    # joining files with concat and read_csv
    data_concat = pd.concat(map(pd.read_csv, merged), ignore_index=True)

    time_end = time.time()
    # print("AIS : ", len(ais_concat['szSrcID'].unique()), "LTE : ", len(lte_concat['szSrcID'].unique()))
    data_processed = preprocessData(data_concat)


    print("Done with ", time_end-time_start)
    return data_processed


def filterShip(data_processed):
    #Data Filtering

    print("Filtering shipments...")
    '''shipment filtering'''
    time_start = time.time()


    #filter ships out where dSOG is nearly 0
    data_valid = data_processed.groupby('szSrcID').filter(lambda group : group['dSOG'].max()>2)

    time_end = time.time()
    # print("AIS : ", len(ais_move['szSrcID'].unique()), "LTE : ", len(lte_move['szSrcID'].unique()))
    print("Done with ", time_end-time_start)
    return data_valid

def get_grid(data, grid_space = 5):
    # Constants
    grid_spacing = grid_space  # 5 km separation
    earth_radius = 6371  # Earth radius in kilometers

    # Calculate the grid size in degrees
    grid_size_deg = (grid_spacing / earth_radius) * (180 / np.pi)

    # Create a grid of latitude and longitude points
    min_lat, max_lat = data['dLat'].min(), data['dLat'].max()
    min_lon, max_lon = data['dLon'].min(), data['dLon'].max()

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

def bound_filter(data, grid, bound_indices):
    lat_grid, lon_grid = grid[0], grid[1]
    lat_indices, lon_indices = bound_indices[0], bound_indices[1]
    data['visited'] = np.logical_or.reduce([data['dLat'].between(lat_grid[lat_indices[0]], lat_grid[lat_indices[1]]) & data['dLon'].between(lon_grid[lon_indices[0]], lon_grid[lon_indices[1]])])
    filtered = data.groupby('szSrcID').filter(lambda group : group['visited'].any())
    return filtered


def mostVisited(data, grid_space = 5, num_points =3):
    # Constants
    grid_spacing = grid_space  # 5 km separation
    earth_radius = 6371  # Earth radius in kilometers

    # Calculate the grid size in degrees
    grid_size_deg = (grid_spacing / earth_radius) * (180 / np.pi)

    # Create a grid of latitude and longitude points
    min_lat, max_lat = data['dLat'].min(), data['dLat'].max()
    min_lon, max_lon = data['dLon'].min(), data['dLon'].max()

    lat_grid = np.arange(min_lat, max_lat, grid_size_deg)
    lon_grid = np.arange(min_lon, max_lon, grid_size_deg)
    heatmap = np.zeros((len(lat_grid) - 1, len(lon_grid) - 1))  # Initialize the heatmap grid

    heatmap, _, _ = np.histogram2d(data['dLat'], data['dLon'], bins=[lat_grid, lon_grid])

     # Change N to the desired number of top grid cells
    top_N_indices = np.argpartition(heatmap, -num_points, axis=None)[-num_points:]
    top_N_indices = np.unravel_index(top_N_indices, heatmap.shape)
    return top_N_indices, [lat_grid, lon_grid]


def gridFilter(data, top_N_indices, grid):
    lat_grid = grid[0]
    lon_grid = grid[1]
    for k, (i, j) in  enumerate(zip(*top_N_indices)):
        data['visited_'+str(k)] = np.logical_or.reduce([data['dLat'].between(lat_grid[i-1], lat_grid[i+1]) & data['dLon'].between(lon_grid[j-1], lon_grid[j+1])])
    
    res = []
    for k in range(len(top_N_indices[0])):
        filtered = data.groupby('szSrcID').filter(lambda group : group['visited_'+str(k)].any())
        res.append(filtered)
    return res


def find_index (data):
    data.reset_index(drop=True)
    try:
        data = data.sort_values('szMsgSendDT').reset_index(drop=True)
        false_filter = data[data['visited']==False]
        true_filter = data[data['visited']==True]
        idx_1 = data[data.szMsgSendDT == false_filter.szMsgSendDT.min()].index.values[0]
        idx_2 = data[data.szMsgSendDT == false_filter.szMsgSendDT.max()].index.values[0]
        idx_3 =data[data.szMsgSendDT == true_filter.szMsgSendDT.min()].index.values[0]
        idx_4 =data[data.szMsgSendDT == true_filter.szMsgSendDT.max()].index.values[0]
        mid = int(len(data)/2)
        if idx_3 > mid : res = data[mid:idx_3]
        else: res = data[idx_3:mid]
        return res
    except: return None



'''
Input
trajA : trajectory data (lat, lon, cog) of one ship (np.array of n x 3)
trajB : trajectory data (lat, lon, cog) of one ship (np.array of n x 3)
k1 : weight of spatial distance
k2 : weight of directional distance

Output
dist : combination distance of spatial and directional distance

'''
def similarity(trajA, trajB, k1 = 0.5, k2 =0.25, k3 = 0.25, m = False):
    # k = [0.5, 0.5]
    # k1, k2 = k[0], k[1]

    cogA = [k for k in trajA.str[2].values if not pd.isna(k)]
    cogB = [l for l in trajB.str[2].values if not pd.isna(l)]
    m_cog =  abs(np.average(cogA)-np.average(cogB)) #mean cog value
    v_cog = abs(np.var(cogA)-np.var(cogB)) #var cog value

    sogA = [k for k in trajA.str[3].values if not pd.isna(k)]
    sogB = [l for l in trajB.str[3].values if not pd.isna(l)]
    m_sog =  abs(np.average(sogA)-np.average(sogB)) #mean sog value
    v_sog = abs(np.var(sogA)-np.var(sogB)) #std sog value

    x = [tuple(i) for i in trajA.str[0:2].values if not isinstance(i, float)]
    y = [tuple(j) for j in trajB.str[0:2].values if not isinstance(j, float)]

    dist = max(directed_hausdorff(x, y)[0], directed_hausdorff(y, x)[0])
    # total_dist = dist    
    # total_dist = k1*dist+k2*m_cog
    # total_dist = k1*dist+k2*v_cog
    # total_dist = k1*dist+k2*m_cog+k3*v_cog

    # total_dist = k1*dist+k2*m_sog
    # total_dist = k1*dist+k2*v_sog
    # total_dist = k1*dist+k2*m_sog+k3*v_sog

    # total_dist = k1*dist+k2*m_sog+k3*m_cog
    # total_dist = k1*dist+k2*v_sog+k3*v_cog

    total_dist = k1*dist+k2*m_sog+k3*v_cog
    # total_dist = k1*dist+k2*v_sog+k3*m_cog

    if m : total_dist = dtw(x,y)
    return total_dist

def sim_mat(data, method, mat_path):
    print("Calculating distances with {}...".format(method))
    if method == 'dtw' : m = True

    time_start = time.time()
    if not os.path.exists(mat_path):
      n = len(data.columns)
      dist_mat = np.zeros((n,n))
      for col in range(n):
        for col2 in range(n):
            val = data[col]
            val2 = data[col2]
            dist = similarity(val, val2, m=m)
            dist_mat[col][col2] = dist
            dist_mat[col2][col] = dist_mat[col][col2]
      pd.DataFrame(dist_mat).to_csv(mat_path, index=False, header=False)

    else:
      dist_mat = np.loadtxt(mat_path, delimiter=',')

    time_end = time.time()
    print("Done with ", time_end-time_start)

    return dist_mat
            
'''
Input
data : total trajectory data (n points x # of vessels )

Output
data with clustered label

'''
def clustering(dist_mat, num_clusters):
    time_start = time.time()
    print("Start Clustering...")
    # Perform agglomerative clustering.
    # The affinity is precomputed (since the distance are precalculated).
    # Use an 'average' linkage. Use any other apart from  'ward'.
    model = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='average')
    
    # Use the distance matrix directly.
    label = model.fit_predict(dist_mat)
    time_end = time.time()
    print("Done with ", time_end-time_start)
    return label

def matching(id_map, label, data):
   cluster = dict()
   for id, assign in enumerate(label):
      shipID = id_map[id]
      cluster[shipID] = assign
   data['cluster'] = data['szSrcID'].map(cluster)
   return data


def plot_scatter(group, color, view = True):
    group = group.sort_values('szMsgSendDT').reset_index(drop=True)
    plt.scatter(group["dLon"], group["dLat"], color = color, s=3)
    if view:
        sp_x, sp_y = group['dLon'][0], group['dLat'][0]
        lp_x, lp_y = group['dLon'][len(group)-1], group['dLat'][len(group)-1]
        plt.text(x=sp_x, y=sp_y, s='start_point', color = color)
        plt.text(x=lp_x, y=lp_y, s='end_point', color = color)


def final(data, mat_path, N, method):
    cols = ['dLon_tr', 'dLat_tr', 'dCOG_tr', 'dSOG_tr']
    ship_list = list(data['szSrcID'].unique())
    print(len(ship_list))
    trajDF = pd.DataFrame()
    i = 0
    print("Generating traj DF")
    time_start = time.time()
    id_map = dict()
    for shipID in ship_list:
        id_map[i] = shipID
        route = data[data['szSrcID']==shipID].copy()
        pos = list(route[cols].values)
        trajDF = pd.concat((trajDF, pd.Series(pos, name = i)), axis = 1)
        i += 1
    time_end = time.time()
    print("Done with ", time_end-time_start)
    dist_mat = sim_mat(trajDF, method, mat_path+"{}.csv".format(len(ship_list)))
    label = clustering(dist_mat, N)
    res_data = matching(id_map, label, data)
    return res_data


