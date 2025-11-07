#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script to verify the PyTorch PDE solver works
"""
import numpy as np
import torch
import sys
import os

# Add the parent directory to path to import torchpde
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torchpde

def test_basic_functionality():
    """Test basic functionality of the converted PyTorch PDE solver"""
    
    print("Testing PyTorch PDE Solver...")
    print("-" * 50)
    
    # Check PyTorch and GPU availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print("-" * 50)
    
    # Simple test case: 1D Heat equation
    # PDE: u_t - 0.1*u_xx = 0
    # IC: u(0, x) = sin(pi*x)
    # BC: u(t, 0) = u(t, 1) = 0 (Dirichlet)
    # Domain: t ∈ [0, 0.1], x ∈ [0, 1]
    
    print("Setting up 1D Heat Equation test...")
    
    # Neural Network Hyperparameters
    NN_parameters = {
        'Network_Type': 'Regular',
        'input_neurons': 2,
        'output_neurons': 1,
        'num_layers': 3,
        'num_neurons': 20
    }
    
    # Neural PDE Hyperparameters
    NPDE_parameters = {
        'Sampling_Method': 'Initial',
        'N_initial': 50,
        'N_boundary': 50,
        'N_domain': 500
    }
    
    # PDE Parameters
    PDE_parameters = {
        'Inputs': 't, x',
        'Outputs': 'u',
        'Equation': 'D(u, t) - 0.1*D2(u, x)',
        'lower_range': [0.0, 0.0],
        'upper_range': [0.1, 1.0],
        'Boundary_Condition': "Dirichlet",
        'Boundary_Vals': None,
        'Initial_Condition': lambda x: np.sin(np.pi*x),
        'Initial_Vals': None
    }
    
    # Generate training data
    N_i = NPDE_parameters['N_initial']
    N_b = NPDE_parameters['N_boundary']
    N_f = NPDE_parameters['N_domain']
    
    lb = np.array(PDE_parameters['lower_range'])
    ub = np.array(PDE_parameters['upper_range'])
    
    # Initial condition points
    X_i = torchpde.sampler.initial_sampler(N_i, lb, ub)
    u_i = PDE_parameters['Initial_Condition'](X_i[:, 1:2])
    
    # Boundary condition points
    X_b = torchpde.sampler.boundary_sampler(N_b, lb, ub)
    u_b = np.zeros((N_b, 1))  # Dirichlet BC: u=0 at boundaries
    
    # Domain points
    X_f = torchpde.sampler.domain_sampler(N_f, lb, ub)
    
    training_data = {
        'X_i': X_i,
        'u_i': u_i,
        'X_b': X_b,
        'u_b': u_b,
        'X_f': X_f
    }
    
    try:
        # Setup model
        print("Creating model...")
        model = torchpde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)
        print("✓ Model created successfully")
        
        # Test training with Adam
        print("\nTraining with Adam optimizer...")
        train_config = {
            'Optimizer': 'adam',
            'learning_rate': 0.001,
            'Iterations': 100  # Just a few iterations for testing
        }
        
        time_adam = model.train(train_config, training_data)
        print(f"✓ Adam training completed in {time_adam:.2f} seconds")
        
        # Test prediction
        print("\nTesting prediction...")
        test_points = np.random.rand(10, 2)
        test_points[:, 0] *= 0.1  # Scale t to [0, 0.1]
        predictions = model.predict(test_points)
        print(f"✓ Predictions shape: {predictions.shape}")
        print(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        print("\n" + "="*50)
        print("All tests passed successfully!")
        print("PyTorch PDE solver is working correctly.")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
