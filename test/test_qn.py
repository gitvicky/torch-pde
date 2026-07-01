#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Quasi-Newton methods for the PyTorch PDE solver
"""

import numpy as np
import torch
import sys
import os
from scipy.optimize import minimize

# Add the torch_pde directory to the path
sys.path.insert(0, os.path.abspath('.'))

import torch_pde
import torch_pde.qnw as qnw

def test_quasi_newton():
    """Test Quasi-Newton methods (L-BFGS) for the PyTorch PDE solver"""
    
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
    
    # Test prediction before training
    test_input = np.array([[0.0, 0.5], [0.5, 0.5]], dtype=np.float32)
    prediction_before = model.predict(test_input)
    print(f"Predictions before QN training: {prediction_before}")
    
    # Test Quasi-Newton training using L-BFGS
    print("Testing Quasi-Newton (L-BFGS) training...")
    
    # Get the loss function from the model
    loss_func = model.loss_func
    
    # Create the SciPy wrapper
    try:
        scipy_optimizer = qnw.Scipy_Keras_Wrapper(model, loss_func, 
                                                   training_data['X_i'], training_data['u_i'],
                                                   training_data['X_b'], training_data['u_b'],
                                                   training_data['X_f'])
        
        # Get initial parameters as 1D array
        initial_params = torch.cat([p.view(-1) for p in model.parameters()])
        
        print(f"Initial params shape: {initial_params.shape}")
        print(f"Number of parameters: {initial_params.numel()}")
        
        # Debug: print parameter shapes
        print("Parameter shapes:")
        for i, p in enumerate(model.parameters()):
            print(f"  Param {i}: {p.shape} ({p.numel()} elements)")
        
        # Run L-BFGS optimization
        result = minimize(
            scipy_optimizer, 
            initial_params.detach().numpy(), 
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 10}
        )
        
        print(f"QN optimization completed: {result.message}")
        print(f"Final loss: {result.fun:.6e}")
        
        # Test prediction after QN training
        prediction_after = model.predict(test_input)
        print(f"Predictions after QN training: {prediction_after}")
        
        # Check if predictions changed (indicating learning)
        predictions_changed = not np.allclose(prediction_before, prediction_after, atol=1e-4)
        print(f"Predictions changed after QN training: {predictions_changed}")
        
        return predictions_changed
        
    except Exception as e:
        print(f"Error during QN training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing PyTorch PDE solver Quasi-Newton methods...")
    success = test_quasi_newton()
    if success:
        print("Quasi-Newton test passed!")
    else:
        print("Quasi-Newton test failed!")