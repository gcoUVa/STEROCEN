# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:18:23 2022

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

def create_agent_PPO(gamma, obs_size, n_actions, n_hidden_layers=1,
                     n_hidden_channels=90, update_interval=50000,
                     nonlinearity=F.tanh, lambd=0.95):
    # Model instanciation (A3C model)
    model = A3CFFSoftmax(
        obs_size, n_actions, n_hidden_channels=n_hidden_channels,
        n_hidden_layers=n_hidden_layers, nonlinearity=nonlinearity)
    # Optimizer
    opt = chainer.optimizers.Adam(eps=1e-2)
    opt.setup(model)
    
    # Agent (PPO)
    agent = chainerrl.agents.PPO(
        model, opt, update_interval=update_interval, gamma=gamma, lambd=lambd)
    
    # Agent info
    agent_info = ('PPO (' + chr(947) + '=' + str(gamma) + ', ' + chr(955) + '='
                  + str(lambd) + '), ' + str(n_hidden_channels) + 'x' +
                  str(n_hidden_layers) + ' (' + nonlinearity.__name__ + ')')
    
    return agent, agent_info, model, opt
