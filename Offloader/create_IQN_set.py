# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:02:04 2022

@author: Mieszko Ferens
"""

import chainer
import chainerrl
import chainer.functions as F
import chainer.links as L
from chainerrl.agents import iqn

def create_agent_IQN(gamma, exploration_func, epsilon, obs_size, n_actions,
                     n_hidden_channels=90, replay_buffer_capacity=1000000,
                     replay_start_size=100000, target_update_interval=50000,
                     nonlinearity=F.tanh):
    # Q function instanciation
    q_func = iqn.ImplicitQuantileQFunction(
        psi=chainerrl.links.Sequence(
            L.Linear(obs_size, n_hidden_channels), nonlinearity),
        phi=chainerrl.links.Sequence(
            iqn.CosineBasisLinear(64, n_hidden_channels), nonlinearity),
        f=L.Linear(n_hidden_channels, n_actions))
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
    
    # Agent (IQN)
    agent = chainerrl.agents.IQN(
            q_func, opt, replay_buffer, gamma, explorer,
            replay_start_size=replay_start_size,
            target_update_interval=target_update_interval)
    
    # Agent info
    if(type(epsilon) == float or type(epsilon) == int):
        agent_info = ('IQN - ' + 'Constant ' + chr(949) + '=' +
                      str(epsilon) + ' (' + chr(947) + '=' + str(gamma) +
                      '), ' + str(n_hidden_channels) + 'x1' + ' (' +
                      nonlinearity.__name__ + ')')
    else:
        agent_info = ('IQN - ' + 'Linear decay ' + chr(949) + '=' +
                      str(epsilon[0]) + '-' + str(epsilon[1]) + ' in ' +
                      str(epsilon[2]) + ' time steps' + ' (' + chr(947) +
                      '=' + str(gamma) + '), ' + str(n_hidden_channels) + 'x1'
                      + ' (' + nonlinearity.__name__ + ')')
    
    return agent, agent_info, q_func, opt, explorer, replay_buffer
