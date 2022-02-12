# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:44:00 2021

@author: Mieszko Ferens
"""

### Environment for computation offloading

import numpy as np
import gym
from gym import spaces
from itertools import product

from .traffic_generator import traffic_generator
from .core_manager import core_manager
from .parameters import (links, links_rate, links_delay, node_type, node_clock,
                         node_cores, n_nodes, net_nodes, n_vehicles, all_paths,
                         node_comb, apps, app_cost, app_data_in, app_data_out,
                         app_max_delay, app_rate, app_info, estimation_err_var,
                         upper_var_limit, lower_var_limit, time_limit,
                         reserv_limit)

"""
Explanation on implemented discrete event simulator:
    
    This environment simulates a network consisting of four types of nodes
    (cloud, MEC, RSU and vehicle). The cloud node is special as there can only
    be one and it has unlimited resources (no queue although its processing
    speed is still modeled after real cores). Together with the cloud node, the
    MEC and RSU nodes form a network interconnected with links. The vehicle
    nodes connect to this network via a shared wireless link at the RSU.
    Each vehicle node runs a set of defined applications with generate
    petitions that will have to be processed at some node of the network. This
    also means that unless the processing occurs at the same vehicle, the input
    data from the applications has to be sent to the corresponding node and
    once processed, the output data has to return.
    The applications from one vehicle cannot be processed by other vehicles.
    
    At each time step, one petition is processed. At this time the action
    by the interacting agent is used to calculate delays and reserve a time
    slot at a core of the selected node. The time at which each petition
    are randomly generated with an exponential distribution and a mean defined
    for each application. These time are also used at each time step to update
    the queues at all cores (time passes). Finally, an observation is returned
    to the agent containing information on the current state of the cores of
    the network and the parameters of the next petition's application.

In this environment the petitions are generated in the traffic_generator class
and passed to the environment. The module that manages the queues of the cores
is the core_manager class.

Approximations are used for modeling transmision delays and queues. The focus
is on the processing (the cores).
"""

class offload_netEnv(gym.Env):

    def __init__(self):
        # Precision limit (in number of decimals) for numpy.float64 variables
        # used to avoid overflow when operating with times in the simulator
        self.precision_limit = 10 # Numpy.float64 has 15 decimals
        
        # Node type list (used as information for metrics during testing)
        self.node_type = node_type
        
        # Links bitrate (in case it needs to be modified during the simulation)
        self.links_rate = links_rate
        
        # Type of next application
        self.app = 0
        # Origin node of next application (vehicle)
        self.app_origin = 0
        # Origin node shift (multiple vehicles can be defined by one node)
        self.app_shift = 0
        # Time until next application
        self.app_time = 0
        # Cost of processing the next application (clock clycles/bit)
        self.app_cost = 0
        # Input data quantity from the next application (kbits)
        self.app_data_in = 0
        # Output data quantity from the next application (kbits)
        self.app_data_out = 0
        # Maximum tolerable delay for the next application (ms)
        self.app_max_delay = 0
        
        # Reference information for defined applications (maximums)
        self.max_cost = max(app_cost)
        self.max_data_in = max(app_data_in)
        self.max_data_out = max(app_data_out)
        self.max_delay = max(app_max_delay)
        self.app_param_count = 4
        
        # Total estimated delay of current application
        self.total_delay_est = 0
        # Total real delays of all processed application since last time step
        self.total_delays = 0
        self.app_types = 0 # Corresponding application types
        
        # For metrics
        # Total number of terminated applications in this time step.
        self.app_count = 0
        # Number of succesfully processed applications in this time step
        self.success_count = 0
        
        # For monitoring purposes the observation will be kept at all times
        self.obs = 0
        
        # Number of cores in the network (except cloud)
        self.n_cores = 0
        for a in range(n_nodes):
            if(node_cores[a] > 0): # Ignore cloud nodes
                self.n_cores += node_cores[a]
        
        # Total number of vehicles in the network
        self.n_vehicles = n_vehicles
        # Number of vehicles per vehicle node (rounded to the nearest integer)
        self.node_vehicles = 0
        
        # Discrete event simulator traffic generation initialization
        self.traffic_generator = traffic_generator(n_nodes, net_nodes, apps,
                                                   app_cost, app_data_in,
                                                   app_data_out, app_max_delay,
                                                   app_rate, app_info)
        
        # Discrete event simulator core manager initialization
        self.core_manager = core_manager(estimation_err_var, upper_var_limit,
                                         lower_var_limit, time_limit,
                                         reserv_limit)
        
        # The observation space has an element per core in the network (with
        # the exception of vehicles, where only one is observable at a time)
        self.observation_space = spaces.Box(low=0, high=1, shape=(
            self.n_cores + self.app_param_count, 1), dtype=np.float32)
        
        self.action_space = spaces.Discrete(net_nodes + 1)

    def step(self, action):
        
        # For each application, the node that processes it cannot be another
        # vehicle, but the local vehicle (the one that generates the petition)
        # can
        # Translate the action to the correct node number if the agent decides
        # to processes the application locally
        if(action == net_nodes):
            action = self.app_origin - 1
            path = []
        # Prepare path between origin node and the selected processor
        else:
            current_nodes = [self.app_origin, action + 1]
            current_nodes.sort()
            path = all_paths[node_comb.index(current_nodes)]
        
        # For each action one node for processing an application is chosen,
        # reserving one of its cores (might queue)
        
        # Calculate the transmission delay for the application's data (in ms)
        forward_delay = 0
        return_delay = 0
        if path:
            for a in range(len(path)-1):
                link = [path[a], path[a+1]]
                link.sort()
                link_index = links.index(link)
                link_rate = self.links_rate[link_index] # in kbit/s
                link_delay = links_delay[link_index] # in ms
                
                forward_delay += (self.app_data_in/link_rate)*1000 + link_delay
                return_delay += link_delay + (self.app_data_out/link_rate)*1000
        
        # Calculate the processing delay at the node (in ms)
        proc_delay = (self.app_data_in*self.app_cost/node_clock[action])*1000
        
        # Limit the precision of the numbers to prevent overflow
        forward_delay = np.around(forward_delay, self.precision_limit)
        return_delay = np.around(return_delay, self.precision_limit)
        proc_delay = np.around(proc_delay, self.precision_limit)
        
        # Choose the next as soon to be available core from the selected
        # processing node and reserve its core for the required time
        # If the selected processing node is the cloud no reservation is done
        # as it has infinite resources
        # Due to a vehicle node defining multiple vehicles, a translation to
        # the corresponding index is required
        self.total_delay_est = 0
        self.total_delays = np.array([], dtype=np.float32)
        self.app_types = np.array([], dtype=np.int32)
        self.app_count = [0]*len(apps)
        if(action): # Not cloud
            if(action == net_nodes): # Process locally
                node_index = (net_nodes-1 + self.app_shift +
                              (action-net_nodes)*self.node_vehicles)
            else: # Process in network
                node_index = action - 1
            self.total_delay_est = self.core_manager.reserve_with_planning(
                node_index, forward_delay, proc_delay, return_delay, self.app)
        else: # Cloud
            # Calculate the total estimated delay of the application processing
            self.total_delay_est = forward_delay + proc_delay + return_delay
            self.total_delays = np.append(
                self.total_delays, self.total_delay_est)
            self.app_types = np.append(self.app_types, self.app)
        
        ## Reward calculation (a priori)
        if(action and self.total_delay_est < 0): # Application not queued
            reward = -1000
            self.app_count[self.app-1] += 1
        else: # Application queued/processed
            reward = 0
        
        # Get next arriving petition
        next_petition = self.traffic_generator.gen_traffic()
        # Assign variables from petition
        self.app = next_petition[0]
        self.app_origin = next_petition[1]
        self.app_shift = next_petition[2]
        self.app_time = np.around(next_petition[3], self.precision_limit)
        self.app_cost = next_petition[4]
        self.app_data_in = next_petition[5] # in kb
        self.app_data_out = next_petition[6] # in kb
        self.app_max_delay = next_petition[7]
        
        ## Observation calculation
        # Calculate the next relevant vehicle index
        vehicle_index = (net_nodes-1 + self.app_shift +
                      (self.app_origin-net_nodes-1)*self.node_vehicles)
        
        # Core and processed applications information
        self.obs, total_delays, app_types = (
            self.core_manager.update_and_calc_obs(
                self.app_time, self.precision_limit, vehicle_index))
        
        self.total_delays = np.append(self.total_delays, total_delays)
        self.app_types = np.append(self.app_types, app_types)
        
        ## Reward calculation (a posteriori)
        # Check for processed and unprocessed applications
        processed = np.where(self.total_delays >= 0)[0]
        failed = np.where(self.total_delays < 0)[0]
        # Find corresponding max tolerable delays
        max_delays = np.array([], dtype=np.float32)
        for i in self.app_types:
            max_delays = np.append(max_delays, app_max_delay[i-1])
            self.app_count[i-1] += 1
        # Calculate total reward for current time step
        remaining = np.clip(
            max_delays[processed] - self.total_delays[processed], None, 0)
        reward += np.sum(remaining)
        reward -= len(self.total_delays[failed]) * 1000
        
        self.success_count = len(np.where(remaining >= 0)[0]) # For metrics
        
        # Add current petition's application type to observation
        app_type = []
        app_type.append(self.app_cost/self.max_cost)
        app_type.append(self.app_data_in/self.max_data_in)
        app_type.append(self.app_data_out/self.max_data_out)
        app_type.append(self.app_max_delay/self.max_delay)
        self.obs = np.append(self.obs, np.array(app_type, dtype=np.float32))

        done = False # This environment is continuous and is never done

        return np.array([self.obs, reward, done, ""], dtype=object)

    def reset(self):
        
        # Define the number of vehicles per vehicle node (rounded to the
        # nearest integer) - Even distribution
        self.node_vehicles = round(self.n_vehicles/self.node_type.count(4))
        
        # Override the bitrate of the shared wireless links between RSUs (type
        # 3 nodes) with vehicles (type 4 nodes) using an approximation
        # Find vehicle nodes and non-vehicle nodes
        RSUs = [node+1 for node, n_type in enumerate(node_type) if n_type == 3]
        vehicles = (
            [node+1 for node, n_type in enumerate(node_type) if n_type == 4])
        # Calculate all possible combinations
        shared = product(RSUs, vehicles)
        shared = list(map(lambda x : list(x), node_comb))
        # Update the only the existings links
        for i in shared:
            i.sort()
            if(i in links):
                # Worst case scenario (with limit)
                self.links_rate[links.index(i)] = (
                    links_rate[links.index(i)]/5)
        
        # Reset and create all cores
        self.core_manager.reset(n_nodes, node_cores, self.node_vehicles,
                                self.node_type)
        
        # Generate initial petitions to get things going
        self.traffic_generator.gen_initial_traffic(self.node_vehicles)
        
        # Get first petition
        next_petition = self.traffic_generator.gen_traffic()
        # Assign variables from petition
        self.app = next_petition[0]
        self.app_origin = next_petition[1]
        self.app_shift = next_petition[2]
        self.app_time = np.around(next_petition[3], self.precision_limit)
        self.app_cost = next_petition[4]
        self.app_data_in = next_petition[5]
        self.app_data_out = next_petition[6]
        self.app_max_delay = next_petition[7]
        
        # Calculate observation
        self.obs = np.array(
            [0]*(self.n_cores + self.app_param_count), dtype=np.float32)
        
        return self.obs

    def render(self, mode='human'):
        # Print current core reservation times
        print('Core reservation time:', self.obs[0:self.n_cores])
        # Print next application to be processed
        print('Next application:', self.app)
    
    def set_total_vehicles(self, total_vehicles):
        self.n_vehicles = total_vehicles
    
    def set_error_var(self, error_var):
        self.core_manager.set_error_var(error_var)

