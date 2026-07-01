#!/usr/bin/env python3

import numpy as np
import torch
import sys
import os

# Add the torch_pde module to the path
sys.path.append('/Users/Vicky/Documents/UKAEA/Code/PINNs/torch-pde')

from torch_pde.network import Network
from torch_pde.pde import PDE
from torch_pde.training_ground import TrainingGround
from torch_pde import boundary_conditions

def test_gradient_computation():
    """Test gradient computation in PDE evaluation"""
    print("Testing gradient computation...")
    
    # Simple test case
    layers = [2, 10, 1]  # 2 inputs (t, x), 10 hidden, 1 output
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    activation = "tanh"
    initializer = "Glorot Uniform"
    
    # Create network
    net = Network(layers, lb, ub, activation, initializer)
    model = net.initialize_NN()
    
    # Create PDE
    eqn_str = "D(u, t) + D(u, x)"  # Simple advection equation
    in_vars = "t, x"
    out_vars = "u"
    pde = PDE(eqn_str, in_vars, out_vars)
    pde.model = model
    
    # Test data
    X = torch.tensor([[0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    
    print(f"Input X: {X}")
    print(f"X.requires_grad: {X.requires_grad}")
    
    # Test model prediction
    u = model(X)
    print(f"Model output u: {u}")
    print(f"u.requires_grad: {u.requires_grad}")
    
    # Test derivative computation
    try:
        derivatives = pde._compute_derivatives(u, X)
        print("Derivatives computed successfully:")
        for key, value in derivatives.items():
            print(f"  {key}: {value}")
            
        # Test PDE evaluation
        pde_result = pde.func(X)
        print(f"PDE evaluation result: {pde_result}")
        
    except Exception as e:
        print(f"Error in gradient computation: {e}")
        import traceback
        traceback.print_exc()

def test_training_loop():
    """Test the training loop with a simple PDE"""
    print("\nTesting training loop...")
    
    # Training configuration
    layers = [2, 10, 1]
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    activation = "tanh"
    initializer = "Glorot Uniform"
    BC = "Dirichlet"
    BC_Vals = None
    N_f = 100
    network_type = "Regular"
    pde_func = None
    eqn_str = "D(u, t) + D(u, x)"
    in_vars = "t, x"
    out_vars = "u"
    sampler = "Initial"
    
    try:
        # Create training ground
        tg = TrainingGround(layers, lb, ub, activation, initializer, BC, BC_Vals, N_f, network_type, pde_func, eqn_str, in_vars, out_vars, sampler)
        
        # Simple training data
        X_i = np.array([[0.0, x] for x in np.linspace(0, 1, 10)])
        u_i = np.array([[np.sin(np.pi * x)] for x in np.linspace(0, 1, 10)])
        X_b = np.array([[t, 0.0] for t in np.linspace(0, 1, 10)] + [[t, 1.0] for t in np.linspace(0, 1, 10)])
        u_b = np.array([[0.0] for _ in range(20)])
        X_f = np.random.uniform(0, 1, (100, 2))
        
        train_data = {
            'X_i': X_i,
            'u_i': u_i,
            'X_b': X_b,
            'u_b': u_b,
            'X_f': X_f
        }
        
        train_config = {
            'Optimizer': 'Adam',
            'learning_rate': 0.01,
            'Iterations': 10
        }
        
        # Test loss computation
        loss = tg.loss_func(
            torch.tensor(X_i, dtype=torch.float32, requires_grad=True),
            torch.tensor(u_i, dtype=torch.float32),
            torch.tensor(X_b, dtype=torch.float32, requires_grad=True),
            torch.tensor(u_b, dtype=torch.float32),
            torch.tensor(X_f, dtype=torch.float32, requires_grad=True)
        )
        
        print(f"Loss computed: {loss}")
        
        # Test training
        print("Starting training...")
        training_time = tg.train(train_config, train_data)
        print(f"Training completed in {training_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in training loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gradient_computation()
    test_training_loop()