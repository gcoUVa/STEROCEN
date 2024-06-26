# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:34:35 2021

@author: Mieszko Ferens
"""

import os
import chainerrl
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime
from pathlib import Path
from operator import add

from agent_creator import make_training_agents
from heuristic_algorithms import make_heuristic_agents
from graph_creator import (makeFigurePlot, makeFigureHistSingle,
                           makeFigureHistSubplot)

optimal_reward = None # Optimal reward of offloader

### - Computation offloading agents with ChainerRL - ###

def train_scenario(env, agents):
    """
    The training environment consists of a network with four types of nodes
    (cloud, MEC, RSU and vehicles). They each have resources in the form of
    cores with different processing speeds, and that have a limited queue size.
    Each vehicle runs a set of application which generate petitions that will
    have to be processed at some node.
    The agent receives information on the reservation time of the queues (from
    0% to 100%) and information on the parameters of the application that
    currently needs to be processed (relative to the others).
    The task is to select a node at which to process the application to ensure
    that its output data is returned to the corresponding vehicle within a
    maximum latency.
    """
    # Create the directory (if not created) where the data will be stored
    results_path = "Results/SingleTraining/" + str(date.today()) + "/"
    i = 0
    while(1):
        if(not os.path.exists(results_path + str(i) + "/")):
            results_path += str(i) + "/"
            break
        i += 1
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    # Create log file for storing the data
    try:
        log_file = open(results_path + "TestLog_" + str(date.today()) + '.txt',
                        'wt', encoding='utf-8')
    except:
        raise KeyboardInterrupt('Error while initializing log file...aborting')
    
    # Add basic information to log file
    log_file.write("Experiment Log - " + str(datetime.today()) + '\n\n')
    log_file.write("Network topology: " + str(env.topology_label) + "\n")
    log_file.write("Vehicles in network: " + str(env.n_vehicles) + "\n")
    log_file.write("Error variance: " + str(env.core_manager.error_var) + "\n")
    log_file.write("---------------------------------------------------\n\n")
    
    # Training
    print('---TRAINING---')
    log_file.write('---TRAINING---\n')
    # Number of time steps to assume a stationary state in the network
    start_up = 1000
    n_time_steps = 2010000 # For 10^-3 precision -> ~10^5 sample points
    # Number of last episodes to use for average reward calculation
    averaging_window = 10000
    x_axis = range(1, start_up+n_time_steps+1) # X axis for ploting results
    # Stores average trining times for each type of agent
    average_total_training_times = []
    average_agent_training_times = []
    # Stores accumulated average rewards of best performing agent in each batch
    top_agents_average = []
    # Iterate through the different types of agents
    for batch in range(len(agents)):
        print('--Batch', batch, 'in training...')
        log_file.write('--Batch ' + str(batch) + ' in training...\n')
        # Stores training times of agents of given type (of batch)
        total_training_times = []
        agent_training_times = []
        # Stores accumulated average rewards during training (all agents)
        average_reward_values = []
        # Iterate through the replicas of agents
        for a in range(len(agents[batch])):
            print('--Agent', a, 'in training...')
            log_file.write('--Agent ' + str(a) + ' in training...\n')
            # Stores rewards during training (one agent)
            rewards = []
            # Stores accumulated averaged rewards during training (one agent)
            average_rewards = []
            # Stores the time taken by the agent to process all time steps
            training_times = 0
            obs = env.reset() # Initialize environment for agent
            reward = 0 # Reward on time step
            done = False
            t = 0 # Time step
            time0 = time.time() # Training starts
            while not done and t < start_up + n_time_steps:
                time_agent = time.time() # Agent starts to process
                # Index 0 is agent object
                action = agents[batch][a][0].act_and_train(obs, reward)
                # Count the time the agents takes to process
                training_times += time.time() - time_agent
                obs, reward, done, _ = env.step(action) # Environment
                rewards.append(reward) # Store time step reward
                t += 1
                # Calculate and store the average reward after max time steps
                if(len(rewards) <= averaging_window):
                    average_rewards.append(
                        sum(rewards)/len(rewards))
                else: # Discard time steps older than averaging window length
                    average_rewards.append(
                            sum(rewards[t-averaging_window:t])
                            /averaging_window)
                # Show how training progresses
                if t % 100 == 0:
                    if(__name__ == "__main__"):
                        print('Time step', t)
                    log_file.write('Time step ' + str(t) + '\n')
                    if(env.total_delay_est >= 0):
                        if(__name__ == "__main__"):
                            print('Application queued/processed succesfully')
                        log_file.write(
                            'Application queued/processed succesfully\n')
                    else:
                        if(__name__ == "__main__"):
                            print('Failed to queue/process the application')
                        log_file.write(
                            'Failed to queue/process the application\n')
                    render_info = env.render()
                    if(__name__ == "__main__"):
                        print(render_info)
                    log_file.write(render_info + '\n')
            
            # End of training
            
            # Store elapsed time during training
            total_training_times.append(time.time() - time0)
            
            # Store elapsed time for agent's processing
            agent_training_times.append(training_times)
            
            # Store accumulated rewards of trained agent
            average_reward_values.append(average_rewards)
            
            # Look for the best performing agent
            if(a == 0):
                best = sum(average_reward_values[a])
                best_agent = a
            elif(best <= sum(average_reward_values[a])):
                best = sum(average_reward_values[a])
                best_agent = a
        
        # Store average training time of agent type (of batch)
        average_total_training_times.append(
            sum(total_training_times)/len(agents[batch]))
        
        # Store average processing time of agent type (of batch)
        average_agent_training_times.append(
            sum(agent_training_times)/len(agents[batch]))
        
        top_agents_average.append(average_reward_values[best_agent])
        
        """
        NOTE: For displaying agent information of a certain type (batch),
              second dimension index can be any existing agent (0 always works)
              and third dimension index 1 is agent info.
        """
        
        # Plot results of batch (average rewards)
        labels = ['Time step', 'Average reward',
                  'Evolution of rewards (' + agents[batch][0][1] + ')']
        makeFigurePlot(
            x_axis, average_reward_values, optimal_reward, labels)
        plt.savefig(results_path + labels[2] + '.svg')
    
    # Plot results of best performing agents (average rewards)
    labels = ['Time step', 'Average reward',
              'Evolution of rewards (best agents)']
    legend = []
    for a in range(len(top_agents_average)):
        legend.append(agents[a][0][1])
    makeFigurePlot(
        x_axis, top_agents_average, optimal_reward, labels, legend)
    plt.savefig(results_path + labels[2] + '.svg')
    
    plt.close('all') # Close all figures
    
    # Average times
    print("\n--Average agent processing times:")
    log_file.write("\n--Average agent processing times:\n")
    for batch in range(len(agents)):
        print(agents[batch][0][1], ': ', average_agent_training_times[batch],
              's', sep='')
        log_file.write(agents[batch][0][1] + ': ' +
                       str(average_agent_training_times[batch]) + 's\n')
    print("\n--Average training times:")
    log_file.write("\n--Average training times:\n")
    for batch in range(len(agents)):
        print(agents[batch][0][1], ': ', average_total_training_times[batch],
              's', sep='')
        log_file.write(agents[batch][0][1] + ': ' +
                       str(average_total_training_times[batch]) + 's\n')
    print('NOTE: The training time takes into account some data collecting!\n')
    log_file.write(
        'NOTE: The training time takes into account some data collecting!\n')
    
    log_file.close() # Close log file
    
    return {'train_avg_total_times': average_total_training_times,
            'train_avg_agent_times': average_agent_training_times}

def test_scenario(env, agents):
    
    # Create the directory (if not created) where the data will be stored
    results_path = "Results/SingleTesting/" + str(date.today()) + "/"
    i = 0
    while(1):
        if(not os.path.exists(results_path + str(i) + "/")):
            results_path += str(i) + "/"
            break
        i += 1
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    # Create log file for storing the data
    try:
        log_file = open(results_path + "TestLog_" + str(date.today()) + '.txt',
                        'wt', encoding='utf-8')
    except:
        raise KeyboardInterrupt('Error while initializing log file...aborting')
    
    # Add basic information to log file
    log_file.write("Experiment Log - " + str(datetime.today()) + '\n\n')
    log_file.write("Network topology: " + str(env.topology_label) + "\n")
    log_file.write("Vehicles in network: " + str(env.n_vehicles) + "\n")
    log_file.write("Error variance: " + str(env.core_manager.error_var) + "\n")
    log_file.write("---------------------------------------------------\n\n")
    
    # Environment parameters required for testing
    n_apps = len(env.traffic_generator.apps)
    n_nodes = len(env.node_type)
    vehicle_nodes = env.node_type.count(4)
    
    # Testing
    # Testing benefit (sum of rewards normalized by number of steps)
    test_benefit = []
    # Testing average of successfully processed application for each batch
    test_success_rate = []
    test_success_rate_per_app = []
    # Testing average of proccessing distribution on nodes per application
    test_act_distribution = []
    # Testing average of total delay observed per application
    test_app_delay_avg = []
    # Testing total delays of each petition per application
    test_app_delays = []
    print('---TESTING---')
    log_file.write('---TESTING---\n')
    # Number of time steps to assume a stationary state in the network
    start_up = 1000
    n_time_steps = 100000 # For 10^-3 precision -> ~10^5 sample points
    for batch in range(len(agents)):
        batch_success_rate = []
        batch_success_rate_per_app = []
        batch_act_distribution = []
        batch_app_count = []
        batch_app_processed = []
        batch_app_delay_avg = []
        batch_app_delays = []
        batch_benefit = []
        print(agents[batch][0][1], ':', sep='')
        log_file.write(str(agents[batch][0][1]) + ':\n')
        for a in range(len(agents[batch])):
            print('  Replica', a, end=':\n')
            log_file.write('  Replica ' + str(a) + ':\n')
            obs = env.reset()
            done = False
            reward = 0
            rewards = 0
            success_count = [0]*n_apps
            last_app = 0
            t = 0
            act_distribution = np.zeros((n_apps, n_nodes-vehicle_nodes+1),
                                        dtype=np.float32)
            act_count = [0]*n_apps
            app_count = [0]*n_apps
            app_processed = [0]*n_apps
            app_delays = []
            for i in range(n_apps):
                app_delays.append([])
            while not done and t < start_up + n_time_steps:
                last_app = env.app
                action = agents[batch][a][0].act(obs)
                obs, reward, done, _ = env.step(action)
                if(t >= start_up):
                    # Store the accumulated reward during testing
                    rewards += reward
                    
                    # Count the returning applications
                    success_count = list(map(
                        add, success_count, env.success_count))
                    #success_count += env.success_count
                    app_count = list(map(add, app_count, env.app_count))
                    
                    # Count the times a certain node processed a specific app
                    act_distribution[last_app-1][action] += 1
                    act_count[last_app-1] += 1
                    
                    # Store the delay of processed applications in the last
                    # time step and count them
                    for i in range(n_apps):
                        indexes = np.where(env.app_types == i+1)[0]
                        total_delays = env.total_delays[indexes]
                        indexes = np.where(total_delays >= 0)[0]
                        for j in indexes:
                            app_delays[i].append(total_delays[j])
                        app_processed[i] += len(indexes)
                
                t += 1
            
            # Calculate the benefit
            batch_benefit.append(rewards/n_time_steps)
            
            # Calculate the fraction of successfully processed applications
            batch_success_rate.append(sum(success_count)*100/sum(app_count))
            batch_success_rate_per_app.append(
                [success*100 / total for success, total in zip(
                    success_count, app_count)])
            
            # Calculate the averages of application distribution throughout the
            # processing nodes
            for i in range(n_apps):
                act_distribution[i] = act_distribution[i]*100/act_count[i]
            batch_act_distribution.append(act_distribution)
            
            # Store the application petition count for the last simulation
            batch_app_count.append(app_count)
            
            # Store the processed application count for the last simulation
            batch_app_processed.append(app_processed)
            
            # Calculate and store the average total delay of each application
            temp = []
            for i in range(n_apps):
                if(app_processed[i]):
                    temp.append(sum(app_delays[i])/app_processed[i])
                else:
                    temp.append(0) # Average cannot be calculated
            batch_app_delay_avg.append(temp)
            
            # Store the registered delays for the tested agents
            batch_app_delays.append(app_delays)
            
            # Print results of replica
            print('   -Benefit: ', str(batch_benefit[a]), sep='')
            print('   -Success rate: ', str(batch_success_rate[a]), '%',
                  sep='')
            print('   |-> Apps: ', str(env.traffic_generator.apps), sep='')
            print('   |-> Rate: ', str(batch_success_rate_per_app[a]), '%',
                  sep='')
            print('   -Processed application rate:')
            print('   |-> Apps: ', str(env.traffic_generator.apps), sep='')
            print('   |-> Num.: ', str(batch_app_processed[a]), sep='')
            print('   |-> Rate: ', str(list(
                np.divide(batch_app_processed[a], batch_app_count[a])*100)),
                sep='')
            print('   -Action distribution:')
            for i in range(n_apps):
                print('   |-> App ', (i+1), ':', sep='')
                print('    |-> Nodes: ', str(env.node_type), sep='')
                print('    |-> Dist.: ',
                      str(batch_act_distribution[a][i]), '%', sep='')
            print('   -Total application delay average:')
            print('   |-> Apps:   ', str(env.traffic_generator.apps), sep='')
            print('   |-> Delays: ', str(batch_app_delay_avg[a]), sep='')
            
            # Log results of replica
            log_file.write('   -Benefit: ' + str(batch_benefit[a]) + '\n')
            log_file.write('   -Success rate: ' +
                           str(batch_success_rate[a]) + '%\n')
            log_file.write('   |-> Apps: ' + str(env.traffic_generator.apps) +
                           '\n')
            log_file.write('   |-> Rate: ' + str(batch_success_rate_per_app[a])
                + '%\n')
            log_file.write('   -Processed application rate:\n')
            log_file.write('   |-> Apps: ' + str(env.traffic_generator.apps) +
                           '\n')
            log_file.write('   |-> Num.: ' + str(batch_app_processed[a]) +
                           '\n')
            log_file.write('   |-> Rate: ' + str(list(
                np.divide(batch_app_processed[a], batch_app_count[a])*100)) +
                '\n')
            log_file.write('   -Action distribution:\n')
            for i in range(n_apps):
                log_file.write('   |-> App ' + str(i+1) + ':\n')
                log_file.write('    |-> Nodes: ' + str(env.node_type) + '\n')
                log_file.write('    |-> Dist.: ' +
                               str(batch_act_distribution[a][i]) + '%\n')
            log_file.write('   -Total application delay average:\n')
            log_file.write('   |-> Apps:   ' +
                           str(env.traffic_generator.apps) + '\n')
            log_file.write('   |-> Delays: ' + str(batch_app_delay_avg[a]) +
                           '\n')
        
        # Look for best performing agent (based on success rate)
        best_agent = batch_success_rate.index(max(batch_success_rate))
        
        # Calculate the averages of benefits
        test_benefit.append(batch_benefit[best_agent])
        
        # Calculate the averages of successfully processed applications
        test_success_rate.append(batch_success_rate[best_agent])
        test_success_rate_per_app.append(
            batch_success_rate_per_app[best_agent])
        
        # Store the average total delay per application of best agent of batch
        test_app_delay_avg.append(batch_app_delay_avg[best_agent])
        
        # Store the delay distribution per application of best agent of batch
        test_app_delays.append(batch_app_delays[best_agent])
        
        # Store the averages of application distribution throughout the
        # processing nodes of best agent
        test_act_distribution.append(batch_act_distribution[best_agent])
    
    # Create histogram of delays of each application (only best agents)
    for i in range(n_apps):
        labels = ['Total application delay (ms)', '',
                  env.traffic_generator.app_info[i]]
        legend = []
        y_axis = []
        for batch in range(len(test_act_distribution)):
            legend.append(agents[batch][0][1] + '(best)')
            y_axis.append(test_app_delays[batch][i])
        bins = 100
        max_delay = env.traffic_generator.app_max_delay[i]
        makeFigureHistSubplot(y_axis, bins, labels, legend, max_delay)
        plt.savefig(results_path + labels[2] + '.svg')
    
    plt.close('all') # Close all figures
    log_file.close() # Close log file
    
    return {'test_benefit': test_benefit,
            'test_success_rate': test_success_rate}

if(__name__ == "__main__"):
    ## Environment (using gym)
    # Checking if the environment is already registered is necesary for
    # subsecuent executions
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'offload' in env:
            print('Remove {} from registry'.format(env))
            del gym.envs.registration.registry.env_specs[env]
    del env_dict
    
    env = gym.make('offloading_net:offload-planning-v1')
    env = chainerrl.wrappers.CastObservationToFloat32(env)
    
    ## Agents (using ChainerRL)
    # Discount factors
    # Note that for comparing average Q values, gamma should be equal for
    # all agents because this parameter influences their calculation.
    gammas = 0.995
    
    # Algorithms to be used
    alg = ['DDQN','SARSA','PAL','TRPO']
    
    # Explorations that are to be analized (in algorithms that use them)
    explorators = 'const'
    epsilons = 0.1
    
    # Define the number of replicas
    repetitions = 1
    
    # Create agents
    agents = make_training_agents(
            env, gammas, explorators, epsilons, alg, repetitions)
    
    # Add heuristic algorithims imitating RL agents to the training for
    # benchmarks
    heuristic_agents = make_heuristic_agents(env)
    for i in range(len(heuristic_agents)):
        agents.insert(0, [heuristic_agents[i]])
    
    ## Train and test
    train_scenario(env, agents)
    test_scenario(env, agents)

