# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:36:44 2022

@author: Mieszko Ferens
"""

import chainer
import chainerrl
import chainer.functions as F

def create_agent_ACER(gamma, obs_size, n_actions, n_hidden_layers=1,
                      n_hidden_channels=90, replay_buffer_capacity=1000000,
                      replay_start_size=100000, nonlinearity=F.tanh,
                      t_max=50000, beta=0.01):
    # Model instanciation
    model = chainerrl.agents.acer.ACERSeparateModel(
        pi=chainerrl.policies.FCSoftmaxPolicy(
            obs_size, n_actions, n_hidden_layers=n_hidden_layers,
            n_hidden_channels=n_hidden_channels, nonlinearity=nonlinearity,
            last_wscale=1.0),
        q=chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
            obs_size, n_actions, n_hidden_channels=n_hidden_channels,
            n_hidden_layers=n_hidden_layers, nonlinearity=nonlinearity,
            last_wscale=1.0))
    # Optimizer
    opt = chainer.optimizers.Adam(eps=1e-2)
    opt.setup(model)
    # Experience replay
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(
        capacity=replay_buffer_capacity)
    
    # Agent (ACER)
    agent = chainerrl.agents.ACER(
            model, opt, t_max, gamma, replay_buffer, beta=beta,
            replay_start_size=replay_start_size)
    
    # Agent info
    agent_info = ('ACER (' + chr(947) + '=' + str(gamma) + ', ' + chr(946) +
                  '=' + str(beta) + '), ' + str(n_hidden_channels) + 'x' +
                  str(n_hidden_layers) + ' (' + nonlinearity.__name__ + ')')
    
    return agent, agent_info, model, opt, replay_buffer
