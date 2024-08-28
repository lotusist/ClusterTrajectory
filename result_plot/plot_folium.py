import folium
from folium import Element
from folium import plugins
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import pickle

network = 'AIS'
PATH_CROPPED_WAYPOINTS = '_.pkl'

cl=242
pA = '목포항'
pB = '제주항'
with open(PATH_CROPPED_WAYPOINTS, 'rb') as file:
    traj = pickle.load(file)

def plot_folium_scatter(group, view = True, step_size = 10, file_name = '0', method='new_a'): # group is a groupby['szSrcID']
    group = group.sort_values('RECPTN_DT').reset_index(drop=True)

    # set cluster color
    unique_clusters = group[f'cluster_{method}'].unique()

    # Define a list of salient colors
    salient_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
                    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
                    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']

    # Map each cluster to a color
    cluster_colors = {cluster: salient_colors[i % len(salient_colors)] for i, cluster in enumerate(unique_clusters)}

    map = folium.Map(location=[group['LA'].iloc[0], group['LO'].iloc[0]], zoom_start=12)

    # sample the waypoints
    first_row = group.iloc[0:1]
    last_row = group.iloc[-1:]
    group = group[::step_size]
    group = pd.concat([first_row, group, last_row], ignore_index=True)

    # mark the start & end points
    if view: 
        sp_lon, sp_lat = group['LO'][0], group['LA'][0]
        lp_lon, lp_lat = group['LO'][len(group)-1], group['LA'][len(group)-1]

        folium.Marker(
            location= [sp_lat, sp_lon],
            icon= folium.DivIcon(html='<div style="font-size: 16pt;">Start</div>'),
            popup=f"Start of Route: {group['ROUTE_ID'][0]}",
        ).add_to(map)

        folium.Marker(
            location= [lp_lat, lp_lon],
            icon= folium.DivIcon(html='<div style="font-size: 16pt;">End</div>'),
            popup=f"End of Route: {group['ROUTE_ID'][0]}",
        ).add_to(map)

    for i, row in group.iterrows():
        cluster_color = cluster_colors.get(row[f'cluster_{method}'], 'gray') 
        folium.CircleMarker(
        location = [row['LA'], row['LO']],
        color = 'black',
        radius = 8, #if representative else 3,
        weight = 1, #if representative else 2,
        fill = True,
        fill_color = cluster_color,
        fill_opacity = 0.9, #if representative else 0.5,
        popup=f"Route: {row['ROUTE_ID']}, TYPE : {row['TYPE']}, TON : {row['TON']}, Cluster: {row[f'cluster_{method}']}"
        ).add_to(map)

     # Constructing the legend HTML
    legend_html = """
    <div style="position: fixed; 
                 bottom: 50px; left: 50%; transform: translateX(-50%); width: 300px; height: auto; 
                 border:2px solid grey; z-index:9999; font-size:20px;
                 background-color: white; opacity: 0.7">
    &nbsp; <b>Legend</b> <br>
    """

    # Sort clusters by their indices and generate legend content
    for cluster in sorted(cluster_colors.keys()):
        legend_html += f'&nbsp; Cluster {cluster} &nbsp; <i class="fa fa-circle" style="color:{cluster_colors[cluster]}"></i><br>'
    
    legend_html += '</div>'
    
    map.get_root().html.add_child(Element(legend_html))

    map.save(f'./map__{network}_{file_name}_{method}_{cl}_large_clusters.html')


plot_folium_scatter(group=traj, view = False, step_size = 20, file_name = f'{pA}-{pB}_cropped', method='hdf')
plot_folium_scatter(group=traj, view = False, step_size = 20, file_name = f'{pA}-{pB}_cropped', method='dtw')
plot_folium_scatter(group=traj, view = False, step_size = 20, file_name = f'{pA}-{pB}_cropped', method='new_a')
