# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:16:42 2022

@author: Mieszko Ferens
"""

import chainer
import chainerrl
import chainer.functions as F

def create_agent_TRPO(gamma, obs_size, n_actions, n_hidden_layers=1,
                      n_hidden_channels=90, update_interval=50000,
                      nonlinearity=F.tanh, lambd=0.95):
    # Policy instanciation
    policy = chainerrl.policies.FCSoftmaxPolicy(
        obs_size, n_actions, n_hidden_channels=n_hidden_channels,
        n_hidden_layers=n_hidden_layers, nonlinearity=nonlinearity,
        last_wscale=1.0)
    # Value function
    vf = chainerrl.v_functions.FCVFunction(
        obs_size, n_hidden_channels=n_hidden_channels,
        n_hidden_layers=n_hidden_layers, nonlinearity=nonlinearity,
        last_wscale=1.0)
    # Optimizer
    opt = chainer.optimizers.Adam(eps=1e-2)
    opt.setup(vf)
    
    # Agent (TRPO)
    agent = chainerrl.agents.TRPO(
        policy=policy, vf=vf, vf_optimizer=opt, gamma=gamma,
        update_interval=update_interval, lambd=lambd)
    
    # Agent info
    agent_info = ('TRPO' + ' (' + chr(947) + '=' + str(gamma) + ', ' + chr(955)
                  + '=' + str(lambd) + '), ' + str(n_hidden_channels) + 'x' +
                  str(n_hidden_layers) + ' (' + nonlinearity.__name__ + ')')
    
    return agent, agent_info, policy, opt
