# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:36:00 2022

@author: Mieszko Ferens
"""

import chainer
import chainerrl
import chainer.functions as F

class A3CFFSoftmax(chainer.ChainList, chainerrl.agents.a3c.A3CModel):

    def __init__(self, obs_size, n_actions, n_hidden_channels=90,
                 n_hidden_layers=1, nonlinearity=F.tanh):
        self.pi = chainerrl.policies.SoftmaxPolicy(
            model=chainerrl.links.MLP(
                obs_size, n_actions,
                tuple([n_hidden_channels]*n_hidden_layers),
                nonlinearity=nonlinearity, last_wscale=1.0))
        self.v = chainerrl.links.MLP(
            obs_size, 1,
            hidden_sizes=tuple([n_hidden_channels]*n_hidden_layers),
            nonlinearity=nonlinearity, last_wscale=1.0)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)

def create_agent_A3C(gamma, obs_size, n_actions, n_hidden_layers=1,
                     n_hidden_channels=90, nonlinearity=F.tanh, t_max=50000,
                     beta=0.01):
    # Model instanciation
    model = A3CFFSoftmax(
        obs_size, n_actions, n_hidden_channels=n_hidden_channels,
        n_hidden_layers=n_hidden_layers, nonlinearity=nonlinearity)
    # Optimizer
    opt = chainer.optimizers.Adam(eps=1e-2)
    opt.setup(model)
    
    # Agent (A3C)
    agent = chainerrl.agents.A3C(model, opt, t_max, gamma, beta=beta)
    
    # Agent info
    agent_info = ('A3C (' + chr(947) + '=' + str(gamma) + ', ' + chr(946) + '='
                  + str(beta) + '), ' + str(n_hidden_channels) + 'x' +
                  str(n_hidden_layers) + ' (' + nonlinearity.__name__ + ')')
    
    return agent, agent_info, model, opt
