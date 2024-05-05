# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:03:18 2022

@author: Mieszko Ferens
"""

import chainer
import chainerrl
import chainer.functions as F
import chainerrl.q_functions as Q

def create_agent_TD3(gamma, exploration_func, epsilon, obs_size, n_actions,
                     n_hidden_layers=1, n_hidden_channels=90,
                     replay_buffer_capacity=1000000, replay_start_size=100000,
                     update_interval=50000, nonlinearity=F.tanh):
    # Policy instanciation
    policy = chainerrl.policies.FCSoftmaxPolicy(
        obs_size, n_actions, n_hidden_channels=n_hidden_channels,
        n_hidden_layers=n_hidden_layers, nonlinearity=nonlinearity,
        last_wscale=1.0)
    # Q function 1 instanciation
    q_func1 = Q.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions, n_hidden_channels=n_hidden_channels,
        n_hidden_layers=n_hidden_layers, nonlinearity=nonlinearity,
        last_wscale=1.0)
    # Q function 2 instanciation
    q_func2 = Q.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions, n_hidden_channels=n_hidden_channels,
        n_hidden_layers=n_hidden_layers, nonlinearity=nonlinearity,
        last_wscale=1.0)
    # Optimizer
    policy_opt = chainer.optimizers.Adam(eps=1e-2)
    policy_opt.setup(policy)
    q1_opt = chainer.optimizers.Adam(eps=1e-2)
    q1_opt.setup(q_func1)
    q2_opt = chainer.optimizers.Adam(eps=1e-2)
    q2_opt.setup(q_func2)
    # Exploration
    if(type(epsilon) == float or type(epsilon) == int):
        explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            epsilon=epsilon, random_action_func=exploration_func)
    else:
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=epsilon[0], end_epsilon=epsilon[1],
        decay_steps=epsilon[2], random_action_func=exploration_func)
    # Experience replay
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(
        capacity=replay_buffer_capacity)
    
    # Agent (TD3)
    agent = chainerrl.agents.TD3(
            policy, q_func1, q_func2, policy_opt, q1_opt, q2_opt,
            replay_buffer, gamma, explorer,
            replay_start_size=replay_start_size,
            update_interval=update_interval)
    
    # Agent info
    if(type(epsilon) == float or type(epsilon) == int):
        agent_info = ('TD3 - ' + 'Constant ' + chr(949) + '=' +
                      str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                      '), ' + str(n_hidden_channels) + 'x' +
                      str(n_hidden_layers) + ' (' + nonlinearity.__name__ +
                      ')')
    else:
        agent_info = ('TD3 - ' + 'Linear decay ' + chr(949) + '=' +
                      str(epsilon[0]) + '-' + str(epsilon[1]) + ' in ' +
                      str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                      '=' + str(gamma) + '), ' + str(n_hidden_channels) + 'x' +
                      str(n_hidden_layers) + ' (' + nonlinearity.__name__ +
                      ')')
    
    return (agent, agent_info, policy, q_func1, q_func2, policy_opt, q1_opt,
            q2_opt, explorer, replay_buffer)
