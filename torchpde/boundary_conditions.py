#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:22:29 2020

@author: Vicky

Neural PDE - PyTorch
Module : Boundary Conditions 
"""

import torch
import torch.autograd as autograd


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
    """Dirichlet boundary condition"""
    u_pred = model(X)
    return u - u_pred


def neumann(model, X, f):
    """Neumann boundary condition (Currently only for 1D)"""
    X = X.requires_grad_(True)
    u = model(X)
    u_X = autograd.grad(u.sum(), X, 
                       create_graph=True,
                       retain_graph=True)[0]
    
    return u_X[:, 1:2] - f


def periodic(model, X, f):
    """Periodic boundary condition (Currently for only 1D)"""
    X = X.requires_grad_(True)
    t = X[:, 0:1]
    x = X[:, 1:2]
    n = int(X.shape[0]/2)
    
    inputs = torch.cat([t, x], dim=1)
    u = model(inputs)
    
    u_x = autograd.grad(u.sum(), x,
                       create_graph=True,
                       retain_graph=True)[0]
    
    return (u[:n] - u[n:]) + (u_x[:n] - u_x[n:])
