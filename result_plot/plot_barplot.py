
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import seaborn as sns

LABELED_WAYPOINT = '.'

def calculate_overlap_ratio(data):
    # Calculate Q1, Q3 and IQR for each cluster
    ranges = []
    for cluster_data in data:
        Q1 = np.percentile(cluster_data, 25)
        Q3 = np.percentile(cluster_data, 75)
        IQR = Q3 - Q1
        non_outlier_range = (max(min(cluster_data), Q1 - 1.5 * IQR), min(max(cluster_data), Q3 + 1.5 * IQR))
        ranges.append(non_outlier_range)

    # Calculate overlap ratios
    overlap_ratios = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            overlap = max(0, min(ranges[i][1], ranges[j][1]) - max(ranges[i][0], ranges[j][0]))
            total_range = max(ranges[i][1], ranges[j][1]) - min(ranges[i][0], ranges[j][0])
            if total_range > 0:
                overlap_ratios[i, j] = overlap / total_range
                overlap_ratios[j, i] = overlap_ratios[i, j]

    return overlap_ratios

def show_des(pA, pB, r, cl, method):
    with open(f'{LABELED_WAYPOINT}/{pA}_{pB}_AIS_{r}_cl{cl}_cropped.pkl', 'rb') as file:
        labeled_waypoint = pickle.load(file)
    cluster_df_dict = {} # dataframe
    cluster_dict = {} # decriptive information
    for i, df in labeled_waypoint.groupby(f'cluster_{method}'):
        cluster_id = df[f'cluster_{method}'].iloc[0]
        cluster_df_dict[cluster_id] = df
        cluster_dict[cluster_id] = {'cluster_id' : i,
                                    'route_nums' : df['ROUTE_ID'].nunique(),
                                    'ship_nums' : df['SHIP_ID'].nunique(),
                                    'types' : list(df['TYPE'].unique()),
                                    'tons' : list(df['TON'].unique()),
                                    # 'entr' : list(),
                                    'mean_sogs' : list(),
                                    'mean_cogs' : list()
                                    }
        for j, route in df.groupby('ROUTE_ID'):
            cluster_dict[cluster_id]['mean_sogs'].append(route['SOG'].mean())
            cluster_dict[cluster_id]['mean_cogs'].append(route['COG_norm'].mean())
            # route['RECPTN_DT'].is_monotonic_increasing
            # cluster_dict[cluster_id]['entr'].append(route['waypoint_order'].is_monotonic_increasing)

    clusters_di = []

    for i in range(len(cluster_dict)):
        print(cluster_dict[i])
        clu_di = dict()
        clu_di['tons'] = cluster_dict[i]['tons']
        clu_di['mean_sogs'] = cluster_dict[i]['mean_sogs']
        clu_di['mean_cogs'] = cluster_dict[i]['mean_cogs']

        clusters_di.append(clu_di)

    # return cluster_df_dict, cluster_dict

    # tons와 mean_sogs의 리스트를 추출
    # tons_list = [cluster['tons'] for cluster in clusters_di]
    mean_cogs_list = [cluster['mean_cogs'] for cluster in clusters_di]
    mean_sogs_list = [cluster['mean_sogs'] for cluster in clusters_di]
    # 박스 플롯 그리기
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # cogs 박스 플롯
    sns.boxplot(data=mean_cogs_list, ax=ax[0])
    ax[0].set_title('Mean Normalized-COGs Distribution by Cluster')
    ax[0].set_xlabel('Cluster')
    ax[0].set_ylabel('Mean COG')

    # mean_sogs 박스 플롯
    sns.boxplot(data=mean_sogs_list, ax=ax[1])
    ax[1].set_title('Mean SOGs Distribution by Cluster')
    ax[1].set_xlabel('Cluster')
    ax[1].set_ylabel('Mean SOG (knots)')

    plt.tight_layout()
    plt.show()

    return mean_cogs_list, mean_sogs_list


def main():
    hdf_mean_cogs_list, hdf_mean_sogs_list = show_des('목포항', '제주항', 242, 5, 'hdf')
    hdf_mean_sogs_within_clusters = [np.var(cluster) for cluster in hdf_mean_sogs_list]
    hdf_mean_cogs_within_clusters = [np.var(cluster) for cluster in hdf_mean_cogs_list]
    hdf_mean_cogs_within_clusters = [np.std(cluster) for cluster in hdf_mean_cogs_list]
    overlap_ratios_cogs = calculate_overlap_ratio(hdf_mean_cogs_list)   
    overlap_ratios_sogs = calculate_overlap_ratio(hdf_mean_sogs_list)   

    dtw_mean_cogs_list, dtw_mean_sogs_list = show_des('목포항', '제주항', 242, 5, 'dtw')
    dtw_mean_sogs_within_clusters = [np.var(cluster) for cluster in dtw_mean_sogs_list]
    dtw_mean_sogs_within_clusters = [np.std(cluster) for cluster in dtw_mean_sogs_list]
    dtw_mean_cogs_within_clusters = [np.var(cluster) for cluster in dtw_mean_cogs_list]
    dtw_mean_cogs_within_clusters = [np.std(cluster) for cluster in dtw_mean_cogs_list]
    overlap_ratios_cogs = calculate_overlap_ratio(dtw_mean_cogs_list)
    overlap_ratios_sogs = calculate_overlap_ratio(dtw_mean_sogs_list)

    new_a_mean_cogs_list, new_a_mean_sogs_list= show_des('목포항', '제주항', 242, 5, 'new_a')
    new_mean_sogs_within_clusters = [np.var(cluster) for cluster in new_a_mean_sogs_list]
    new_mean_sogs_within_clusters = [np.std(cluster) for cluster in new_a_mean_sogs_list]
    new_mean_cogs_within_clusters = [np.var(cluster) for cluster in new_a_mean_cogs_list]
    new_mean_cogs_within_clusters = [np.std(cluster) for cluster in new_a_mean_cogs_list]
    overlap_ratios_sogs = calculate_overlap_ratio(new_a_mean_sogs_list)
    overlap_ratios_cogs = calculate_overlap_ratio(new_a_mean_cogs_list)
    
if __name__ == '__main__':
    main()