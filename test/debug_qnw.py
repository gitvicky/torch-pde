#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test to debug the QNW module
"""

import numpy as np
import torch
import sys
import os

# Add the torch_pde directory to the path
sys.path.insert(0, os.path.abspath('.'))

import torch_pde
import torch_pde.qnw as qnw

def debug_qnw():
    """Debug the QNW module step by step"""
    
    # Simple test parameters for a basic PDE
    NN_parameters = {
        'Network_Type': 'Regular',
        'input_neurons': 2,
        'output_neurons': 1,
        'num_layers': 2,
        'num_neurons': 10
    }
    
    NPDE_parameters = {
        'Sampling_Method': 'Initial',
        'N_initial': 10,
        'N_boundary': 20,
        'N_domain': 50
    }
    
    PDE_parameters = {
        'Inputs': 't, x',
        'Outputs': 'u',
        'Equation': 'D(u, t) + D(u, x)',  # Simple advection equation
        'lower_range': [0.0, 0.0],
        'upper_range': [1.0, 1.0],
        'Boundary_Condition': "Dirichlet",
        'Boundary_Vals': None,
        'Initial_Condition': lambda x: np.sin(np.pi * x),
        'Initial_Vals': None
    }
    
    print("Setting up model...")
    model = torch_pde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)
    print("Model setup successful!")
    
    # Create training data
    print("Creating training data...")
    
    # Initial condition data
    X_i = np.random.rand(NPDE_parameters['N_initial'], 2)
    X_i[:, 0] = 0.0  # t=0
    u_i = np.sin(np.pi * X_i[:, 1:2])
    
    # Boundary condition data
    X_b = np.random.rand(NPDE_parameters['N_boundary'], 2)
    X_b[:NPDE_parameters['N_boundary']//2, 0] = 0.0
    X_b[NPDE_parameters['N_boundary']//2:, 0] = 1.0
    u_b = np.zeros_like(X_b[:, 0:1])
    
    # Domain data
    X_f = np.random.rand(NPDE_parameters['N_domain'], 2)
    
    # Get the loss function from the model
    loss_func = model.loss_func
    
    # Test the QNW wrapper creation
    print("Creating QNW wrapper...")
    try:
        scipy_optimizer = qnw.Scipy_Keras_Wrapper(model, loss_func, 
                                                   X_i, u_i, X_b, u_b, X_f)
        print("QNW wrapper created successfully!")
        
        # Test parameter extraction
        print("Testing parameter extraction...")
        shapes = [p.shape for p in model.parameters()]
        print(f"Parameter shapes: {shapes}")
        
        initial_params = torch.cat([p.view(-1) for p in model.parameters()])
        print(f"Initial params shape: {initial_params.shape}")
        print(f"Number of parameters: {initial_params.numel()}")
        
        # Test the val_and_grads_1d function
        print("Testing val_and_grads_1d function...")
        loss_value, grads = scipy_optimizer(initial_params)
        print(f"Loss: {loss_value:.6e}")
        print(f"Gradients shape: {grads.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error during QNW testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Debugging QNW module...")
    success = debug_qnw()
    if success:
        print("QNW debug test passed!")
    else:
        print("QNW debug test failed!")