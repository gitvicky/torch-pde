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
        
        self.lb = lb 
        self.ub = ub 
        
        self.activation_name = activation
        self.activation = self.get_activation(activation)
        self.initializer = self.get_initializer(initializer)
        
    def get_activation(self, name):
        """Get PyTorch activation function by name"""
        activations = {
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "relu": F.relu,
            "leaky_relu": F.leaky_relu
        }
        return activations.get(name, torch.tanh)
    
    def get_initializer(self, name):
        """Get PyTorch initializer by name"""
        initializers = {
            "Glorot Uniform": nn.init.xavier_uniform_,
            "Glorot Normal": nn.init.xavier_normal_,
            "Random Normal": nn.init.normal_,
            "Random Uniform": nn.init.uniform_,
            "Truncated Normal": lambda x: nn.init.normal_(x, mean=0, std=0.02),
            "Variance Scaling": nn.init.kaiming_normal_,
            "Constant": lambda x: nn.init.constant_(x, 1),
            "Zero": nn.init.zeros_
        }
        return initializers.get(name, nn.init.xavier_uniform_)
    
    def initialize_NN(self):
        """ Initialises a fully connected deep neural network """
        layers_list = []
        
        # Input layer
        layers_list.append(nn.Linear(self.layers[0], self.layers[1]))
        
        # Hidden layers
        for ii in range(2, len(self.layers) - 1):
            layers_list.append(nn.Linear(self.layers[ii-1], self.layers[ii]))
        
        # Output layer
        layers_list.append(nn.Linear(self.layers[-2], self.layers[-1]))
        
        model = nn.Sequential(*layers_list)
        
        # Apply initialization
        for layer in model:
            if isinstance(layer, nn.Linear):
                self.initializer(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        return model

    def res_net_block(self, input_data, neurons):
        """ResNet block with skip connections"""
        x = F.linear(input_data, torch.randn(neurons, input_data.size(1)), torch.randn(neurons))
        x = self.activation(x)
        x = F.linear(x, torch.randn(neurons, neurons), torch.randn(neurons))
        x = self.activation(x)
        x = torch.cat([x, input_data], dim=1)
        x = self.activation(x)
        return x
        
    def initialize_resnet(self, num_blocks):
        """ Initialises a Resnet """
        class ResNet(nn.Module):
            def __init__(self):
                super(ResNet, self).__init__()
                self.input_layer = nn.Linear(self.layers[0], self.neurons)
                self.res_blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.neurons, self.neurons),
                        self.activation,
                        nn.Linear(self.neurons, self.neurons),
                        self.activation
                    ) for _ in range(num_blocks)
                ])
                self.output_layer = nn.Linear(self.neurons, self.num_outputs)
                
                # Initialize weights
                self._initialize_weights()
                
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        self.initializer(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            
            def forward(self, x):
                x = self.activation(self.input_layer(x))
                for block in self.res_blocks:
                    residual = x
                    x = block(x)
                    x = x + residual
                x = self.output_layer(x)
                return x
        
        return ResNet()

    def normalise(self, X):
        """ Performs Min-Max Normalisation on the input parameters using the predefined lower and upper ranges """
        # Convert numpy arrays to tensors if needed
        if isinstance(self.lb, np.ndarray):
            lb_tensor = torch.tensor(self.lb, dtype=X.dtype, device=X.device)
        else:
            lb_tensor = self.lb
            
        if isinstance(self.ub, np.ndarray):
            ub_tensor = torch.tensor(self.ub, dtype=X.dtype, device=X.device)
        else:
            ub_tensor = self.ub
            
        return 2.0*(X - lb_tensor)/(ub_tensor - lb_tensor) - 1.0
     
    def forward(self, model, X):
        """ Performs the Feedforward Operation """
        X_norm = self.normalise(X)
        return model(X_norm)