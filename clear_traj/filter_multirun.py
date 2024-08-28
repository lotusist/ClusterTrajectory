import subprocess, multiprocessing, sys, os
import json

PORT_BOUND_JSON = '_.json'


port_polygons = json.load(open(PORT_BOUND_JSON))
port_names = list(port_polygons.keys())

py_file = 'filter_1.py'
ports_list = [['목포항', '제주항'], 
              ['부산항', '제주항'], 
              ['인천항', '제주항']]

####################################################
def run(cmd): 
    subprocess.run(cmd, shell=True)

####################################################

num_cpu = multiprocessing.cpu_count()
cmds = []
for ports in ports_list:
    if (ports[0] not in port_names) or (ports[1] not in port_names):
        print(f'{ports[0]} or {ports[1]} NOT IN THE PORT_POLYGON_LIST.')
    cmds += f'[time python {py_file} {ports[0]} {ports[1]}]'

p = multiprocessing.Pool(processes=int(num_cpu))
p.map(run, cmds)

