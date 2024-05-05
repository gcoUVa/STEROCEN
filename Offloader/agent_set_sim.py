# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:42:30 2022

@author: Mieszko Ferens
"""

import gym
import chainerrl
import os
import chainer.functions as F

from create_DQN_set import create_agent_DQN
from create_DDQN_set import create_agent_DDQN
from create_CategoricalDQN_set import create_agent_CategoricalDQN
from create_CategoricalDDQN_set import create_agent_CategoricalDDQN
from create_ResidualDQN_set import create_agent_ResidualDQN
from create_IQN_set import create_agent_IQN
from create_DoubleIQN_set import create_agent_DoubleIQN
from create_DPP_set import create_agent_DPP
from create_SARSA_set import create_agent_SARSA
from create_AL_set import create_agent_AL
from create_PAL_set import create_agent_PAL
from create_DoublePAL_set import create_agent_DoublePAL
from create_PCL_set import create_agent_PCL
from create_REINFORCE_set import create_agent_REINFORCE
from create_SAC_set import create_agent_SAC
from create_TD3_set import create_agent_TD3
from create_TRPO_set import create_agent_TRPO
from create_A3C_set import create_agent_A3C
from create_PPO_set import create_agent_PPO
from create_ACER_set import create_agent_ACER

"""
There are four more algorithms implemented in ChainerRL that are not being used
here: A2C, DDPG, NSQ and PGT. Due to the lack of examples that are applicable
to the discrete action offloading scenario they are omitted here.
"""

from offloader import train_scenario, test_scenario

path_to_env = "../Environments/offloading-net/offloading_net/envs/"

## Setup simulation state in temporal file (used for creating the appropriate
##environment, i.e. using the correct network topology)

"""
 --- Defined topologies ---
 Topologies:
     1. network_branchless 
     2. network_branchless_v2
     3. network_branched

 Topology labels:
     1. Branchless network
     2. Branchless network v2
     3. Branched network

 --------------------------
"""

topologies = ["network_branchless", "network_branchless_v2",
              "network_branched"]
topology_labels = ["Branchless network", "Branchless network v2",
                   "Branched network"]

top_index = 1 # Pick one index out of the above

## Define what is the current network topology for simulation in state file
try:
    state_file = open(path_to_env + "net_topology", 'wt')
except:
    raise KeyboardInterrupt(
        'Error while initializing state file...aborting')

state_file.write(topologies[top_index])
state_file.close()

# Checking if the environment is already registered is necessary for
# subsecuent executions
env_dict = gym.envs.registration.registry.env_specs.copy()
for envs in env_dict:
    if 'offload' in envs:
        print('Remove {} from registry'.format(envs))
        del gym.envs.registration.registry.env_specs[envs]
del env_dict

## Environment (using gym)
env = gym.make('offloading_net:offload-planning-v1')
env = chainerrl.wrappers.CastObservationToFloat32(env)

#env.set_random_seed(100)

# Remove temporary file so it's not read in other simulations
if(os.path.exists(path_to_env + "net_topology")):
    os.remove(path_to_env + "net_topology")

# Set some parameters for the environment
env.set_error_var(1) # Coeficient of variation of the gaussian of the noise
env.set_upper_var_limit(1) # Upper limit of noise (relative to noiseless time)
env.set_lower_var_limit(0) # Lower limit of noise (relative to noiseless time)
env.set_total_vehicles(50)

### Agents (using ChainerRL)
agents = []

# Environment specific parameters
exploration_func = env.action_space.sample
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
"""
## --- DQN agents ----------------

# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create DQN agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_DQN(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- DDQN agents ---------------

# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create DDQN agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_DDQN(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- Categorical DQN agents ----

# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create Categorical DQN agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_CategoricalDQN(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- Categorical DDQN agents ---

# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create Categorical DDQN agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_CategoricalDDQN(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- Residual DQN agents -------

# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create Residual DQN agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_ResidualDQN(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- IQN agents ----------------

# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 1

# Create IQN agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_IQN(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- DoubleIQN agents ----------

# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 1

# Create DoubleIQN agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_DoubleIQN(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## --------------------------------
## --- DPP agents -----------------

# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Positive constant
eta = [0.8]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create DPP agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_DPP(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j], eta=eta[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- SARSA agents --------------

# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create SARSA agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_SARSA(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
"""
## --- AL agents -----------------

# Discount factors
gammas = [0.5]

# Weight of advantages
alpha = [0.9]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create AL agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_AL(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j], alpha=alpha[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- PAL agents ----------------

# Discount factors
gammas = [0.5]

# Weight of advantages
alpha = [0.9]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create PAL agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_PAL(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j], alpha=alpha[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- DoublePAL agents ----------

# Discount factors
gammas = [0.5]

# Weight of advantages
alpha = [0.9]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
target_update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create DoublePAL agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_DoublePAL(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            target_update_interval=target_update_interval[j],
            nonlinearity=nonlinearity[j], alpha=alpha[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- PCL agents ----------------

# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]

# Define the number of replicas
repetitions = 2 # TODO (note to self: it can overflow the RAM)

# Create PCL agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_PCL(
            gammas[j], exploration_func, epsilons[j], obs_size,
            n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- REINFORCE agents ----------

# Weight coefficient fro the entropy regularization term
beta = [0.01]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Define the number of replicas
repetitions = 2 # NOTE: Algorithm takes a lot of memory

# Create REINFORCE agents
for j in range(len(beta)): # For example go through beta
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_REINFORCE(
            obs_size, n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            nonlinearity=nonlinearity[j], beta=beta[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- SAC agents ----------------
# TODO (doesn't work)
# Discount factors
gammas = [0.5]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create SAC agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_SAC(
            gammas[j], obs_size, n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            update_interval=update_interval[j], nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- TD3 agents ----------------
# TODO (doesn't work)
# Discount factors
gammas = [0.5]

# Explorations that are to be analized
epsilons = [[0.25, 0.05, 250000]]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]
update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create TD3 agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_TD3(
            gammas[j], exploration_func, epsilons[j], obs_size, n_actions,
            n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            update_interval=update_interval[j], nonlinearity=nonlinearity[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- TRPO agents ---------------

# Discount factors
gammas = [0.5]

# Lambda-return factor
lambd = [1]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Policy update interval
update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create TRPO agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_TRPO(
            gammas[j], obs_size, n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            update_interval=update_interval[j], nonlinearity=nonlinearity[j],
            lambd=lambd[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- A3C agents ----------------

# Discount factors
gammas = [0.5]

# Weight coefficient for the entropy regularization term
beta = [0.01]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Model update interval
t_max = [25000]

# Define the number of replicas
repetitions = 4

# Create A3C agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_A3C(
            gammas[j], obs_size, n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            nonlinearity=nonlinearity[j], t_max=t_max[j], beta=beta[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------
## --- PPO agents ----------------

# Discount factors
gammas = [0.5]

# Lambda-return factor
lambd = [1]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Model update interval
update_interval = [25000]

# Define the number of replicas
repetitions = 4

# Create PPO agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_PPO(
            gammas[j], obs_size, n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            update_interval=update_interval[j], nonlinearity=nonlinearity[j],
            lambd=lambd[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## --------------------------------
## --- ACER agents ----------------
# TODO (it doesn't work)
# Discount factors
gammas = [0.5]

# Weight coefficient for the entropy regularization term
beta = [0.01]

# Neural network parameters
n_hidden_layers = [1]
n_hidden_channels = [60]
nonlinearity = [F.relu]

# Experience replay parameters
replay_buffer_capacity = [500000]
replay_start_size = [50000]

# Model update interval
t_max = [25000]

# Define the number of replicas
repetitions = 4

# Create ACER agents
for j in range(len(gammas)): # For example go through gammas
    agent_replicas = []
    for k in range(repetitions):
        agent_replicas.append(create_agent_ACER(
            gammas[j], obs_size, n_actions, n_hidden_layers=n_hidden_layers[j],
            n_hidden_channels=n_hidden_channels[j],
            replay_buffer_capacity=replay_buffer_capacity[j],
            replay_start_size=replay_start_size[j],
            nonlinearity=nonlinearity[j], t_max=t_max[j], beta=beta[j]))
    agents.append(agent_replicas)
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)
raise KeyboardInterrupt("end")
## -------------------------------

## Launch simulations
train_results = train_scenario(env, agents)
test_results = test_scenario(env, agents)