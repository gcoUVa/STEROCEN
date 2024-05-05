# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:06:47 2022

@author: Mieszko Ferens
"""

import chainer
import chainerrl
import chainer.functions as F

def create_agent_REINFORCE(obs_size, n_actions, n_hidden_layers=1,
                           n_hidden_channels=90, update_interval=50000,
                           nonlinearity=F.tanh, beta=0.01):
    # Policy instanciation
    model = chainerrl.policies.FCSoftmaxPolicy(
        obs_size, n_actions, n_hidden_channels=n_hidden_channels,
        n_hidden_layers=n_hidden_layers, nonlinearity=nonlinearity,
        last_wscale=1.0)
    # Optimizer
    opt = chainer.optimizers.Adam(eps=1e-2)
    opt.setup(model)
    
    # Agent (REINFORCE)
    agent = chainerrl.agents.REINFORCE(model, opt, beta=beta)
    
    # Agent info
    agent_info = ('REINFORCE' + ' (' + chr(946) + '=' + str(beta) + '), ' +
                  str(n_hidden_channels) + 'x' + str(n_hidden_layers) + ' (' +
                  nonlinearity.__name__ + ')')
    
    return agent, agent_info, model, opt
