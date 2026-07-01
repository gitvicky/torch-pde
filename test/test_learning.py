#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to verify that the PyTorch PDE solver is actually learning
"""

import numpy as np
import torch
import sys
import os

# Add the torch_pde directory to the path
sys.path.insert(0, os.path.abspath('.'))

import torch_pde

def test_learning():
    """Test that the model actually learns something during training"""
    
    # Simple test parameters for a basic PDE
    NN_parameters = {
        'Network_Type': 'Regular',
        'input_neurons': 2,
        'output_neurons': 1,
        'num_layers': 3,
        'num_neurons': 20
    }
    
    NPDE_parameters = {
        'Sampling_Method': 'Initial',
        'N_initial': 50,
        'N_boundary': 100,
        'N_domain': 200
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
    test_input = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]], dtype=np.float32)
    prediction_before = model.predict(test_input)
    print(f"Predictions before training: {prediction_before}")
    
    # Test training with more iterations
    print("Testing training...")
    train_config = {
        'Optimizer': 'adam',
        'learning_rate': 0.01, 
        'Iterations': 100
    }
    
    training_time = model.train(train_config, training_data)
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Test prediction after training
    prediction_after = model.predict(test_input)
    print(f"Predictions after training: {prediction_after}")
    
    # Check if predictions changed (indicating learning)
    predictions_changed = not np.allclose(prediction_before, prediction_after, atol=1e-4)
    print(f"Predictions changed after training: {predictions_changed}")
    
    # Test specific points to see if they make sense
    # At t=0, should be close to initial condition sin(pi*x)
    test_initial = np.array([[0.0, 0.5]], dtype=np.float32)
    pred_initial = model.predict(test_initial)
    expected_initial = np.sin(np.pi * 0.5)  # sin(pi*0.5) = 1
    print(f"Prediction at t=0, x=0.5: {pred_initial[0][0]:.4f}, Expected: {expected_initial:.4f}")
    
    return predictions_changed

if __name__ == "__main__":
    print("Testing PyTorch PDE solver learning...")
    success = test_learning()
    if success:
        print("Learning test passed!")
    else:
        print("Learning test failed - predictions didn't change significantly")