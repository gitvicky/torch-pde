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

def test_comprehensive_training():
    """Test comprehensive training with different PDEs and configurations"""
    print("Testing comprehensive training scenarios...")
    
    # Test different PDEs
    pde_cases = [
        {
            "name": "Advection Equation",
            "eqn_str": "D(u, t) + D(u, x)",
            "in_vars": "t, x",
            "out_vars": "u",
            "expected_behavior": "Should learn advection pattern"
        },
        {
            "name": "Burgers Equation", 
            "eqn_str": "D(u, t) + u*D(u, x)",
            "in_vars": "t, x",
            "out_vars": "u",
            "expected_behavior": "Should learn shock wave pattern"
        },
        {
            "name": "Heat Equation",
            "eqn_str": "D(u, t) - D2(u, x)",
            "in_vars": "t, x", 
            "out_vars": "u",
            "expected_behavior": "Should learn diffusion pattern"
        }
    ]
    
    for case in pde_cases:
        print(f"\n--- Testing {case['name']} ---")
        print(f"Equation: {case['eqn_str']}")
        print(f"Expected: {case['expected_behavior']}")
        
        # Training configuration
        layers = [2, 20, 1]
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 1.0])
        activation = "tanh"
        initializer = "Glorot Uniform"
        BC = "Dirichlet"
        BC_Vals = None
        N_f = 200
        network_type = "Regular"
        sampler = "Initial"
        
        try:
            # Create training ground
            tg = TrainingGround(layers, lb, ub, activation, initializer, BC, BC_Vals, N_f, 
                              network_type, None, case['eqn_str'], case['in_vars'], case['out_vars'], sampler)
            
            # Create training data - simple sine wave initial condition
            x_vals = np.linspace(0, 1, 20)
            X_i = np.array([[0.0, x] for x in x_vals])
            u_i = np.array([[np.sin(np.pi * x)] for x in x_vals])
            
            # Boundary conditions - zero at boundaries
            t_vals = np.linspace(0, 1, 15)
            X_b = np.array([[t, 0.0] for t in t_vals] + [[t, 1.0] for t in t_vals])
            u_b = np.array([[0.0] for _ in range(len(t_vals) * 2)])
            
            # Domain points
            X_f = np.random.uniform(0, 1, (N_f, 2))
            
            train_data = {
                'X_i': X_i,
                'u_i': u_i,
                'X_b': X_b,
                'u_b': u_b,
                'X_f': X_f
            }
            
            # Test with Adam optimizer
            train_config_adam = {
                'Optimizer': 'Adam',
                'learning_rate': 0.01,
                'Iterations': 50
            }
            
            print("Training with Adam optimizer...")
            training_time = tg.train(train_config_adam, train_data)
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Test prediction
            test_points = np.array([[0.25, 0.5], [0.5, 0.5], [0.75, 0.5]])
            predictions = tg.predict(test_points)
            print(f"Predictions at test points: {predictions.flatten()}")
            
            # Test with L-BFGS optimizer
            print("Training with L-BFGS optimizer...")
            train_config_lbfgs = {
                'Optimizer': 'L-BFGS',
                'learning_rate': 0.1,
                'Iterations': 20
            }
            
            training_time_lbfgs = tg.train(train_config_lbfgs, train_data)
            print(f"L-BFGS training completed in {training_time_lbfgs:.2f} seconds")
            
            # Test prediction after L-BFGS
            predictions_lbfgs = tg.predict(test_points)
            print(f"L-BFGS predictions at test points: {predictions_lbfgs.flatten()}")
            
            print(f"✅ {case['name']} test completed successfully")
            
        except Exception as e:
            print(f"❌ Error in {case['name']} test: {e}")
            import traceback
            traceback.print_exc()

def test_pde_parsing():
    """Test PDE parsing functionality"""
    print("\n--- Testing PDE Parsing ---")
    
    # Test cases for PDE parsing
    test_cases = [
        "D(u, t) + D(u, x)",
        "D(u, t) + u*D(u, x)", 
        "D(u, t) - D2(u, x)",
        "D2(u, t) + D2(u, x) + u",
        "D(u, t) + D(u, x) + D(u, y)"
    ]
    
    for eqn_str in test_cases:
        print(f"\nParsing: {eqn_str}")
        try:
            pde = PDE(eqn_str, "t, x", "u")
            print(f"  Has higher order derivatives: {pde.has_higher_order_derivatives}")
            print(f"  Parsed terms: {pde.terms}")
            print("✅ Parsing successful")
        except Exception as e:
            print(f"❌ Parsing failed: {e}")

if __name__ == "__main__":
    test_comprehensive_training()
    test_pde_parsing()