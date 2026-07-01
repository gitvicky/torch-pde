#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the PyTorch version of the PDE solver
"""

import numpy as np
import torch
import sys
import os

# Add the torch_pde directory to the path
sys.path.insert(0, os.path.abspath('.'))

import torch_pde

def test_basic_functionality():
    """Test basic functionality of the PyTorch PDE solver"""
    
    # Simple test parameters
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
    try:
        model = torch_pde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)
        print("Model setup successful!")
        
        # Test basic prediction
        test_input = np.array([[0.5, 0.5]])
        print(f"Test input shape: {test_input.shape}")
        
        prediction = model.predict(test_input)
        print(f"Prediction successful! Output: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"Error during model setup or prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing PyTorch PDE solver...")
    success = test_basic_functionality()
    if success:
        print("Basic functionality test passed!")
    else:
        print("Basic functionality test failed!")