#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:22:29 2020

@author: Vicky

Neural PDE - PyTorch
Module : Boundary Conditions 

"""

import torch

def select(name):
    try: 
        return {
            "Dirichlet": dirichlet,
            "Neumann": neumann,
            "Periodic": periodic
            }[name]
    except KeyError:
        raise KeyError("Unknown Boundary Condition")


def dirichlet(model, X, u):
    u_pred = model(X)
    return u - u_pred


def neumann(model, X, f): #Currently only for 1D
    u = model(X)
    u_X = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    return u_X[:, 1:2] - f


def periodic(model, X, f): # Currently for only 1D
    t = X[:, 0:1]
    x = X[:, 1:2]
    n = int(X.shape[0]/2)
    u =  model(torch.cat([t, x], 1))
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    return (u[:n] - u[n:]) + (u_x[:n] - u_x[:n])