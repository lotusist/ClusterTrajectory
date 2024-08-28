from functions import *
import os
import pandas as pd
import time
import json

LTE_PATH = "_LTEM"
AIS_PATH = "_AIS"
VALID_SHIP_PATH = "_/data_glob"
PORT_BOUND_JSON = '_.json'

def main():

    network = "ais"
    if network == "lte":
        path = LTE_PATH
    else : 
        path = AIS_PATH

    res_path = f"{VALID_SHIP_PATH}/{network}_valid_ship_norm.csv"
    if os.path.exists(res_path):
        data = pd.read_csv(res_path).reset_index()
    else:
        concat = collectData(path)
        data = filterShip(concat)
        data.to_csv(res_path, index = False)

    data = data[['szMsgSendDT', 'szSrcID', 'dLon', 'dLat', 'dCOG', 'dSOG', 'dLon_tr', 'dLat_tr', 'dCOG_tr', 'dSOG_tr']]

    grid_space = 5   #unit : km
    num_points = 3   #top number of points
    num_clusters = 5

    #find most visited points and grid
    _ , grid = mostVisited(data, grid_space, num_points)

    #or given points 
    # top_N_indices = [[list of lat], [list of lon]]

    #n ports(center point) in korea
    ports = list(map(list, zip(*list(json.load(open(PORT_BOUND_JSON)).values())[:3])))
    lat_indices = np.searchsorted(grid[0], ports[1], side="right") - 1
    lon_indices = np.searchsorted(grid[1], ports[0], side="right") - 1

    top_N_indices = (lat_indices, lon_indices)
    dataset = gridFilter(data, top_N_indices, grid)
    

    
    method = 'dtw'
    filtered_dataset = []
    for i in range(len(top_N_indices[0])):
        sample= dataset[i]
        sample.rename(columns = {'visited_'+str(i) : 'visited'}, inplace = True)
        sample = sample.reset_index(drop = True)
        sample = sample.groupby('szSrcID').apply(lambda group : find_index(group))
        filtered_dataset.append(sample)

    res_filtered_dataset=[]
    for i in range(len(top_N_indices[0])):
        mat_path = "./dist_mat/{}_mat_{}_cog_v_sog_m_{}_".format(method, network, i)
        plot_path = "./plot/plot_clustered_{}_cog_v_sog_m_{}_{}.png".format(network, i, num_clusters)
        filtered_data = filtered_dataset[i].reset_index(drop=True)
        res_data = final(filtered_data, mat_path, num_clusters, method)
        res_filtered_dataset.append(res_data)
        location_set = res_data
        print("location : ", i)
        print("mean of COG : ",list(location_set.groupby('cluster').apply(lambda group : np.mean(group['dCOG_tr']))))
        print("variance of COG : ",list(location_set.groupby('cluster').apply(lambda group : np.var(group['dCOG_tr']))))
        print("mean of SOG : ",list(location_set.groupby('cluster').apply(lambda group : np.mean(group['dSOG_tr']))))
        print("variance of SOG : ",list(location_set.groupby('cluster').apply(lambda group : np.var(group['dSOG_tr']))))
        
        colors = ['red', 'blue', 'orange',  'green', 'yellow']
        lat, lon = top_N_indices[0][i], top_N_indices[1][i]
        plt.figure(figsize=(5,5))
        location_set.groupby('szSrcID').apply(lambda group : plot_scatter(group, colors[group['cluster'].unique()[0]]))
        plt.plot([grid[1][lon]],[grid[0][lat]], marker="*", markersize=20, markeredgecolor="none", markerfacecolor="green", zorder = 100)
        plt.savefig(plot_path)
        # plt.xlim(lon_range)
        # plt.ylim(lat_range)

            

if __name__ == '__main__':
    main()

