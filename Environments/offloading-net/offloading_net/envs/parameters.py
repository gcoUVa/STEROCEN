# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:44:33 2021

@author: Mieszko Ferens
"""

import pandas as pd
import networkx as netx
from itertools import product

path_to_env = '../Environments/offloading-net/offloading_net/envs/'

### Parameters for computation offloading in a network environment

## Network topology

topologies = ["network_branchless", "network_branchless_v2",
              "network_branched"]
topology_labels = ["Branchless network", "Branchless network v2",
                   "Branched network"]

# Choose network topology
try:
    state_file = open(path_to_env + "net_topology", "rt")
    topology = state_file.read() # Defined from main
    state_file.close()
except:
    topology = 'network_branchless_v2' # Default for single simulation testing

topology_label = topology_labels[topologies.index(topology)]

print("Environment is being created for network topology: " + topology_label)

# Links and bitrate/delay of each link (loaded from .csv)
data = pd.read_csv(path_to_env + topology + '.csv')
links = data[['original', 'connected']].values.tolist()
links_rate = data['bitrate'].values.tolist()
links_delay = data['delay'].values.tolist()

# Get parameters for nodes in the network (loaded from .csv)
data = pd.read_csv(path_to_env + topology + '_nodes.csv')
node_type = data['type'].values.tolist()
node_clock = data['clock'].values.tolist()
node_cores = data['cores'].values.tolist()
node_buffer = data['buffer'].values.tolist()

# Check that only one cloud node exists
if(node_type.count(1) > 1):
    raise KeyboardInterrupt(
        "Error in network definition: Only one cloud node is allowed!")

# Get definided applications (loaded from .csv)
data = pd.read_csv(path_to_env + topology + '_apps.csv')
apps = data['app'].values.tolist()
app_cost = data['cost'].values.tolist()
app_data_in = data['data_in'].values.tolist()
app_data_out = data['data_out'].values.tolist()
app_max_delay = data['max_delay'].values.tolist()
app_rate = data['rate'].values.tolist()
app_benefit = data['benefit'].values.tolist()
app_info = data['info'].values.tolist()

# Count network nodes (with and without vehicles, marked as type 4) and vehicle
# nodes
n_nodes = len(node_type)
net_nodes = sum(map(lambda x : x<4, node_type))
vehicle_nodes = n_nodes - net_nodes

# Define the default number of total vehicles in the network (each vehicle
# node can represent multiple vehicles)
n_vehicles = 50

# Default variation coeficient of error in estimation of processing time
estimation_err_var = 0
# Limits to variation of times
upper_var_limit = 1 # Percentage out of corresponding total time
lower_var_limit = 0 # Percentage out of corresponding total time

# Define the cores' queue limit reservation amount
reserv_limit = 1000 # Simulator limit

## Precalculated routes

# Definition of network so paths can be calculated
net = netx.Graph()
net.add_edges_from(links)

# Find vehicle nodes and non-vehicle nodes
sources = [node+1 for node, n_type in enumerate(node_type) if n_type == 4]
targets = [node+1 for node, n_type in enumerate(node_type) if n_type != 4]

# Calculate all necessary combinations of nodes (from each vehicle to other)
node_comb = product(sources, targets)
node_comb = list(map(lambda x : list(x), node_comb))
for i in node_comb: i.sort()

# Precalculation of routes
all_paths = []
for pair in node_comb:
    all_paths.append(netx.shortest_path(net, pair[0], pair[1]))

