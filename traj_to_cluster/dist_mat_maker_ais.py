from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import numpy as np
import pickle, os, time
from tslearn.metrics import dtw 
from sklearn.cluster import AgglomerativeClustering


PATH_CROPPED_WAYPOINTS = '_.pkl'
MAT_PATH = './dist_mat/'

with open(PATH_CROPPED_WAYPOINTS, 'rb') as file:
    traj = pickle.load(file)
pA = '목포항'
pB = '제주항'

def similarity(trajA, trajB, method='new', k1 = 0.5, k2 =0.25, k3 = 0.25):
    # k = [0.5, 0.5]
    # k1, k2 = k[0], k[1]

    cogA = [k for k in trajA.str[2].values if not pd.isna(k)]
    cogB = [l for l in trajB.str[2].values if not pd.isna(l)]
    m_cog =  abs(np.average(cogA)-np.average(cogB)) #mean cog value
    v_cog = abs(np.var(cogA)-np.var(cogB)) #var cog value

    sogA = [k for k in trajA.str[3].values if not pd.isna(k)]
    sogB = [l for l in trajB.str[3].values if not pd.isna(l)]
    m_sog =  abs(np.average(sogA)-np.average(sogB)) #mean sog value
    v_sog = abs(np.var(sogA)-np.var(sogB)) #std sog value`

    x = [tuple(i) for i in trajA.str[0:2].values if not isinstance(i, float)]
    y = [tuple(j) for j in trajB.str[0:2].values if not isinstance(j, float)]
    
    time_start_haus = time.time()
    dist = max(directed_hausdorff(x, y)[0], directed_hausdorff(y, x)[0])
    time_end_haus = time.time()

    if method == 'hdf':
      total_dist = dist
    elif method == 'new':
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
    elif method == 'dtw': 
      time_start_dtw = time.time()
      total_dist = dtw(x,y)
      time_end_dtw = time.time()
    else:
       print('input valid method')

    return total_dist

def sim_mat(data, mat_path, method ='new', k1 = 0.5, k2 =0.25, k3 = 0.25, network= 'AIS'):
    print("Calculating distances with {}...".format(method))
    mat_name = mat_path+f"{pA}_{pB}_{network}_{method}_{k1}_{k2}_{k3}_cropped.csv"
    print(mat_name)

    time_start = time.time()
    if not os.path.exists(mat_name):
      n = len(data.columns)
      dist_mat = np.zeros((n,n))
      for col in range(n):
        for col2 in range(n):
            val = data[col]
            val2 = data[col2]
            dist = similarity(val, val2, method, k1, k2, k3)
            dist_mat[col][col2] = dist
            dist_mat[col2][col] = dist_mat[col][col2]
      pd.DataFrame(dist_mat).to_csv(mat_name, index=False, header=False)

    else:
      dist_mat = np.loadtxt(mat_name, delimiter=',')

    time_end = time.time()
    print("Done with sim_mat ", time_end-time_start)

    return dist_mat

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

def matching(id_map, label, data, method='hdf'):
   cluster = dict()
   for id, assign in enumerate(label):
      routeID = id_map[id]
      cluster[routeID] = assign
   data[f'cluster_{method}'] = data['ROUTE_ID'].map(cluster)
   return data

def final(data, N=3, mat_path=MAT_PATH, network = 'AIS'):
    cols = ['LO_norm', 'LA_norm', 'COG_norm', 'SOG_norm']
    if network == 'AIS':
      data = data[data['NETWORK'] == 'AIS'].copy()
    elif network == 'LTEM': 
      data = data[data['NETWORK'] == 'LTEM'].copy()
    
    route_list = list(data['ROUTE_ID'].unique())
    print(len(route_list))

    trajDF = pd.DataFrame()
    i = 0
    print("Generating traj DF")
    time_start = time.time()
    id_map = dict()
    for ID in route_list:
        id_map[i] = ID
        route = data[data['ROUTE_ID']==ID].copy()
        pos = list(route[cols].values)
        trajDF = pd.concat((trajDF, pd.Series(pos, name = i)), axis = 1)
        i += 1
    time_end = time.time()
    print("Done Generating trajDF with ", time_end-time_start)
    hausdorff_dist_mat = sim_mat(trajDF, 
                                 mat_path= mat_path,
                                 method='hdf',  
                                 k1 = 1, k2 =0, k3 = 0, network = network)
    dtw_dist_mat = sim_mat(trajDF, 
                              mat_path= mat_path,
                              method='dtw',  
                              k1 = 1, k2 =0, k3 = 0, network = network)
    new_dist_mat_a = sim_mat(trajDF, 
                              mat_path= mat_path,
                              method='new',  
                              k1 = 0.5, k2 =0.25, k3 = 0.25, network = network)


    label_hausdorff = clustering(hausdorff_dist_mat, N)
    label_dtw = clustering(dtw_dist_mat, N)
    label_new_a = clustering(new_dist_mat_a, N)
    
    res_data = matching(id_map, label_hausdorff, data, method='hdf')
    res_data = matching(id_map, label_dtw, res_data, method='dtw')
    res_data = matching(id_map, label_new_a, res_data, method='new_a')

    return res_data


##########################
final(traj, N=5, mat_path=MAT_PATH, network = 'AIS')
