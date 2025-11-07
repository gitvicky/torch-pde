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
import torch.nn as nn
import torch.optim as optim


# ------------------ OPTIMIZER ------------------------------

def get_optimizer(name, parameters=None, lr=None):
    
    if name in ["sgd", "adam", "adamw", "adagrad", "adadelta", "adamax", "rmsprop"]:
        optimizer_class = {
            "sgd": optim.SGD,
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "adagrad": optim.Adagrad,
            "adadelta": optim.Adadelta,
            "adamax": optim.Adamax,
            "rmsprop": optim.RMSprop,
        }[name]
        
        if parameters is not None:
            if name == "adadelta":
                return optimizer_class(parameters), "GD"
            else:
                return optimizer_class(parameters, lr=lr), "GD"
        else:
            return optimizer_class, "GD"
    
    elif name in ["LBFGS", "L-BFGS"]:
        if parameters is not None:
            return optim.LBFGS(parameters, lr=lr if lr else 1.0, 
                              max_iter=20, max_eval=25,
                              history_size=50,
                              tolerance_grad=1e-5,
                              tolerance_change=1e-9,
                              line_search_fn="strong_wolfe"), "QN_Torch"
        else:
            return optim.LBFGS, "QN_Torch"
    
    else:
        return optimize.minimize, "QN_Scipy"
         
    
    
# ------------------ ACTIVATION FUNCTION ----------------------------

def get_activation(name):
    activations = {
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU()
    }
    
    if name in activations:
        return activations[name]
    else:
        raise ValueError(f"Unknown Activation Function: {name}")
    
    
    
# ------------------ KERNEL INITIALIZER ----------------------------

def get_initializer(name):
    """
    Returns a function that can be used to initialize weights
    """
    def glorot_uniform(tensor):
        nn.init.xavier_uniform_(tensor)
        
    def glorot_normal(tensor):
        nn.init.xavier_normal_(tensor)
        
    def random_normal(tensor):
        nn.init.normal_(tensor, mean=0.0, std=0.02)
        
    def random_uniform(tensor):
        nn.init.uniform_(tensor, a=-0.05, b=0.05)
        
    def truncated_normal(tensor):
        nn.init.trunc_normal_(tensor, mean=0.0, std=0.02)
        
    def constant_init(tensor):
        nn.init.constant_(tensor, val=1.0)
        
    def zero_init(tensor):
        nn.init.zeros_(tensor)
        
    def kaiming_normal(tensor):
        nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity='relu')
        
    def kaiming_uniform(tensor):
        nn.init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity='relu')
    
    initializers = {
        "Glorot Uniform": glorot_uniform,
        "Glorot Normal": glorot_normal,
        "Random Normal": random_normal,
        "Random Uniform": random_uniform,
        "Truncated Normal": truncated_normal,
        "Variance Scaling": kaiming_normal,  # Similar to TF's VarianceScaling
        "Constant": constant_init,
        "Zero": zero_init,
        "Kaiming Normal": kaiming_normal,
        "Kaiming Uniform": kaiming_uniform
    }
    
    if name in initializers:
        return initializers[name]
    else:
        raise ValueError(f"Unknown Initializer: {name}")
