#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
More comprehensive test for the PyTorch PDE solver
"""

import numpy as np
import torch
import sys
import os

# Add the torch_pde directory to the path
sys.path.insert(0, os.path.abspath('.'))

import torch_pde

def test_training_functionality():
    """Test training functionality of the PyTorch PDE solver"""
    
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
        'N_initial': 20,
        'N_boundary': 40,
        'N_domain': 100
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
    try:
        model = torch_pde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)
        print("Model setup successful!")
        
        # Create some simple training data
        print("Creating training data...")
        
        # Initial condition data
        X_i = np.random.rand(NPDE_parameters['N_initial'], 2)
        X_i[:, 0] = 0.0  # t=0
        u_i = np.sin(np.pi * X_i[:, 1:2])
        
        # Boundary condition data
        X_b = np.random.rand(NPDE_parameters['N_boundary'], 2)
        # Set some boundaries to t=0 and some to t=1
        X_b[:NPDE_parameters['N_boundary']//2, 0] = 0.0
        X_b[NPDE_parameters['N_boundary']//2:, 0] = 1.0
        u_b = np.zeros_like(X_b[:, 0:1])
        
        # Domain data
        X_f = np.random.rand(NPDE_parameters['N_domain'], 2)
        
        training_data = {
            'X_i': X_i.astype(np.float32),
            'u_i': u_i.astype(np.float32),
            'X_b': X_b.astype(np.float32),
            'u_b': u_b.astype(np.float32),
            'X_f': X_f.astype(np.float32)
        }
        
        # Test training with a few iterations
        print("Testing training...")
        train_config = {
            'Optimizer': 'adam',
            'learning_rate': 0.01, 
            'Iterations': 10
        }
        
        training_time = model.train(train_config, training_data)
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Test prediction after training
        test_input = np.array([[0.5, 0.5]], dtype=np.float32)
        prediction = model.predict(test_input)
        print(f"Prediction after training: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"Error during training test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing PyTorch PDE solver training functionality...")
    success = test_training_functionality()
    if success:
        print("Training functionality test passed!")
    else:
        print("Training functionality test failed!")