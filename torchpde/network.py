#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:01:56 2020

@author: Vicky

Neural PDE - PyTorch
Module : Network
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)

# Set default tensor type to double precision
torch.set_default_dtype(torch.float64)

from . import options

class Network(object):
    def __init__(self, layers, lb, ub, activation, initializer):
        """
        Parameters
        ----------
        layers : LIST
            Number of neurons in each layer
        lb : ARRAY
            Lower Range of the time and space domain
        ub : ARRAY
            Upper Range of the time and space domain
        activation : STR
            Name of the activation Function
        initializer : STR
            Name of the Initialiser for the neural network weights
        
        Returns
        -------
        None.
        """
        self.layers = layers
        self.num_inputs = layers[0]
        self.num_outputs = layers[-1]
        self.neurons = layers[1]

        self.lb = torch.tensor(lb, dtype=torch.float64)
        self.ub = torch.tensor(ub, dtype=torch.float64)
        
        self.activation_name = activation
        self.activation = options.get_activation(activation)
        self.initializer = options.get_initializer(initializer)
        
        
    def initialize_NN(self):
        """Initialises a fully connected deep neural network"""
        layers_list = []
        
        # Input layer
        layers_list.append(nn.Linear(self.layers[0], self.layers[1]))
        layers_list.append(self.activation)
        
        # Hidden layers
        for ii in range(1, len(self.layers) - 2):
            layers_list.append(nn.Linear(self.layers[ii], self.layers[ii+1]))
            layers_list.append(self.activation)
            
        # Output layer
        layers_list.append(nn.Linear(self.layers[-2], self.layers[-1]))
        
        model = nn.Sequential(*layers_list)
        
        # Apply weight initialization
        for layer in model:
            if isinstance(layer, nn.Linear):
                self.initializer(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        return model

    def res_net_block(self, in_features, out_features):
        """Creates a residual block"""
        block = nn.Sequential(
            nn.Linear(in_features, out_features),
            self.activation,
            nn.Linear(out_features, out_features)
        )
        return block
        
    def initialize_resnet(self, num_blocks):
        """Initialises a ResNet"""
        
        class ResNet(nn.Module):
            def __init__(self, layers, neurons, num_blocks, activation, initializer):
                super(ResNet, self).__init__()
                self.layers_dims = layers
                self.neurons = neurons
                self.num_blocks = num_blocks
                self.activation = activation
                
                # Build ResNet blocks
                self.blocks = nn.ModuleList()
                
                # First layer to expand input to hidden size
                self.input_layer = nn.Linear(layers[0], neurons)
                
                # Residual blocks
                for _ in range(num_blocks):
                    block = nn.Sequential(
                        nn.Linear(neurons, neurons),
                        activation,
                        nn.Linear(neurons, neurons)
                    )
                    self.blocks.append(block)
                
                # Final layers
                self.fc1 = nn.Linear(neurons, neurons)
                self.fc2 = nn.Linear(neurons, layers[-1])
                
                # Initialize weights
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        initializer(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
                
            def forward(self, x):
                x = self.input_layer(x)
                x = self.activation(x)
                
                for block in self.blocks:
                    residual = x
                    x = block(x)
                    x = x + residual
                    x = self.activation(x)
                
                x = self.fc1(x)
                x = self.activation(x)
                x = self.fc2(x)
                return x
        
        model = ResNet(self.layers, self.neurons, num_blocks, 
                      self.activation, self.initializer)
        return model
        
    
    def normalise(self, X):
        """Performs Min-Max Normalisation on the input parameters"""
        X_tensor = X if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float64)
        return 2.0 * (X_tensor - self.lb) / (self.ub - self.lb) - 1.0
    
    def forward(self, model, X):
        """Performs the Feedforward Operation"""
        X_norm = self.normalise(X)
        return model(X_norm)
