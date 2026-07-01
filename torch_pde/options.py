#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:13:13 2020

@author: Vicky

Neural PDE - PyTorch
Module : Options
"""

import numpy as np
from scipy import optimize
import torch
import torch.optim as optim


# ------------------ OPTIMIZER ------------------------------

def get_optimizer(name, lr=None):
    if name in ["sgd", "nadam", "adagrad", "adadelta", "adamax", "adam", "rmsprop"]:
        # Return optimizer class and learning rate separately
        optimizers = {
            "sgd": optim.SGD,
            "nadam": optim.NAdam,
            "adagrad": optim.Adagrad,
            "adadelta": optim.Adadelta,
            "adamax": optim.Adamax,
            "adam": optim.Adam,
            "rmsprop": optim.RMSprop
        }
        return optimizers[name], lr, "GD"
    elif name in ["BFGS", "L-BFGS"]:
        return {
            "BFGS": optimize.minimize,
            "L-BFGS": optimize.minimize,
        }[name], None, "QN_Scipy"
    elif name == "L-BFGS-PyTorch":
        return torch.optim.LBFGS, None, "QN_PyTorch"
    else:
        return optimize.minimize, None, "QN_Scipy"
    
    raise ValueError("Unknown Optimizer")


# ------------------ ACTIVATION FUNCTION ----------------------------

def get_activation(name):
    activations = {
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "relu": torch.relu,
        "leaky_relu": torch.nn.functional.leaky_relu
    }
    return activations.get(name, torch.tanh)
    raise ValueError("Unknown Activation Function")


# ------------------ KERNEL INITIALIZER ----------------------------

def get_initializer(name):
    initializers = {
        "Glorot Uniform": torch.nn.init.xavier_uniform_,
        "Glorot Normal": torch.nn.init.xavier_normal_,
        "Random Normal": torch.nn.init.normal_,
        "Random Uniform": torch.nn.init.uniform_,
        "Truncated Normal": lambda x: torch.nn.init.normal_(x, mean=0, std=0.02),
        "Variance Scaling": torch.nn.init.kaiming_normal_,
        "Constant": lambda x: torch.nn.init.constant_(x, 1),
        "Zero": torch.nn.init.zeros_
    }
    return initializers.get(name, torch.nn.init.xavier_uniform_)
    raise ValueError("Unknown Initializer")