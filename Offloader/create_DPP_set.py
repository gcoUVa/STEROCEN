# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:37:58 2022

@author: Mieszko Ferens
"""

import chainer
import chainerrl
import chainer.functions as F
import chainerrl.q_functions as Q

def create_agent_DPP(gamma, exploration_func, epsilon, obs_size, n_actions,
                     n_hidden_layers=1, n_hidden_channels=90,
                     replay_buffer_capacity=1000000, replay_start_size=100000,
                     target_update_interval=50000, nonlinearity=F.tanh,
                     eta=1.0):
    # Q function instanciation
    q_func = Q.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions, n_hidden_channels=n_hidden_channels,
        n_hidden_layers=n_hidden_layers, nonlinearity=nonlinearity,
        last_wscale=1.0)
    # Optimizer
    opt = chainer.optimizers.Adam(eps=1e-2)
    opt.setup(q_func)
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
    
    # Agent (DPP)
    agent = chainerrl.agents.DPP(
            q_func, opt, replay_buffer, gamma, explorer,
            replay_start_size=replay_start_size,
            target_update_interval=target_update_interval)
    
    # Agent info
    if(type(epsilon) == float or type(epsilon) == int):
        agent_info = ('DPP - ' + 'Constant ' + chr(949) + '=' +
                      str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                      ', ' + chr(951) + '=' + str(eta) + '), ' +
                      str(n_hidden_channels) + 'x' + str(n_hidden_layers) +
                      ' (' + nonlinearity.__name__ + ')')
    else:
        agent_info = ('DPP - ' + 'Linear decay ' + chr(949) + '=' +
                      str(epsilon[0]) + '-' + str(epsilon[1]) + ' in ' +
                      str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                      '=' + str(gamma) + ', ' + chr(951) + '=' + str(eta) +
                      '), ' + str(n_hidden_channels) + 'x' +
                      str(n_hidden_layers) + ' (' + nonlinearity.__name__ +
                      ')')
    
    return agent, agent_info, q_func, opt, explorer, replay_buffer
