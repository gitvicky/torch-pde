#!/usr/bin/env python3

import numpy as np
import torch
import sys
import os
import time

# Add the torch_pde module to the path
sys.path.append('/Users/Vicky/Documents/UKAEA/Code/PINNs/torch-pde')

from torch_pde.training_ground import TrainingGround

def test_original_examples():
    """Test with examples similar to the original TensorFlow implementation"""
    print("=== Testing Original Examples ===")
    
    # Test cases based on original experiments
    test_cases = [
        {
            "name": "Advection Equation - Sine Wave",
            "eqn_str": "D(u, t) + D(u, x)",
            "in_vars": "t, x",
            "out_vars": "u",
            "initial_condition": lambda x: np.sin(np.pi * x),
            "boundary_condition": lambda t: 0.0,
            "expected_behavior": "Wave should propagate to the right"
        },
        {
            "name": "Heat Equation - Sine Wave",
            "eqn_str": "D(u, t) - D2(u, x)",
            "in_vars": "t, x",
            "out_vars": "u",
            "initial_condition": lambda x: np.sin(np.pi * x),
            "boundary_condition": lambda t: 0.0,
            "expected_behavior": "Wave should diffuse and decay"
        },
        {
            "name": "Burgers Equation - Sine Wave",
            "eqn_str": "D(u, t) + u*D(u, x)",
            "in_vars": "t, x",
            "out_vars": "u",
            "initial_condition": lambda x: np.sin(np.pi * x),
            "boundary_condition": lambda t: 0.0,
            "expected_behavior": "Wave should steepen and form shock"
        }
    ]
    
    # Common configuration
    layers = [2, 20, 1]
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    activation = "tanh"
    initializer = "Glorot Uniform"
    BC = "Dirichlet"
    N_f = 1000
    network_type = "Regular"
    sampler = "Initial"
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Equation: {case['eqn_str']}")
        print(f"Expected: {case['expected_behavior']}")
        
        try:
            # Create training ground
            tg = TrainingGround(layers, lb, ub, activation, initializer, BC, None, N_f, 
                              network_type, None, case['eqn_str'], case['in_vars'], case['out_vars'], sampler)
            
            # Generate training data
            x_vals = np.linspace(0, 1, 30)
            X_i = np.array([[0.0, x] for x in x_vals])
            u_i = np.array([[case['initial_condition'](x)] for x in x_vals])
            
            t_vals = np.linspace(0, 1, 20)
            X_b = np.array([[t, 0.0] for t in t_vals] + [[t, 1.0] for t in t_vals])
            u_b = np.array([[case['boundary_condition'](t)] for t in t_vals] + 
                          [[case['boundary_condition'](t)] for t in t_vals])
            
            X_f = np.random.uniform(0, 1, (N_f, 2))
            
            train_data = {
                'X_i': X_i,
                'u_i': u_i,
                'X_b': X_b,
                'u_b': u_b,
                'X_f': X_f
            }
            
            # Test both optimizers
            optimizers = [
                {'name': 'Adam', 'config': {'Optimizer': 'Adam', 'learning_rate': 0.01, 'Iterations': 200}},
                {'name': 'L-BFGS', 'config': {'Optimizer': 'L-BFGS', 'learning_rate': 0.1, 'Iterations': 100}}
            ]
            
            for opt in optimizers:
                print(f"\nTraining with {opt['name']} optimizer...")
                
                # Train the model
                start_time = time.time()
                training_time = tg.train(opt['config'], train_data)
                end_time = time.time()
                
                print(f"Training completed in {training_time:.2f} seconds (wall time: {end_time - start_time:.2f}s)")
                
                # Test predictions on a grid
                test_points = []
                for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    for x in np.linspace(0, 1, 20):
                        test_points.append([t, x])
                
                test_points = np.array(test_points)
                predictions = tg.predict(test_points)
                
                # Analyze results
                print(f"Sample predictions at t=0.5:")
                t_05_indices = [i for i, (t, x) in enumerate(test_points) if abs(t - 0.5) < 1e-6]
                for i, idx in enumerate(t_05_indices[:5]):
                    t, x = test_points[idx]
                    pred = predictions[idx]
                    print(f"  x={x:.2f} -> u={pred[0]:.4f}")
                
                # Test loss computation
                final_loss = tg.loss_func(
                    torch.tensor(X_i, dtype=torch.float32, requires_grad=True),
                    torch.tensor(u_i, dtype=torch.float32),
                    torch.tensor(X_b, dtype=torch.float32, requires_grad=True),
                    torch.tensor(u_b, dtype=torch.float32),
                    torch.tensor(X_f, dtype=torch.float32, requires_grad=True)
                )
                
                print(f"Final loss: {final_loss.item():.6f}")
                
                # Check physical behavior
                initial_max = np.max(np.abs(u_i))
                final_max = np.max(np.abs(predictions[-20:]))  # Last time step
                
                if case['name'].startswith("Advection"):
                    # For advection, the magnitude should be preserved
                    if abs(initial_max - final_max) < 0.2:
                        print("✅ Advection behavior: magnitude preserved")
                    else:
                        print(f"⚠️  Advection behavior: magnitude changed from {initial_max:.3f} to {final_max:.3f}")
                elif case['name'].startswith("Heat"):
                    # For heat equation, the magnitude should decrease
                    if final_max < initial_max * 0.8:
                        print("✅ Heat behavior: magnitude decreased")
                    else:
                        print(f"⚠️  Heat behavior: magnitude changed from {initial_max:.3f} to {final_max:.3f}")
                elif case['name'].startswith("Burgers"):
                    # For Burgers, the solution should change significantly
                    if abs(initial_max - final_max) > 0.1:
                        print("✅ Burgers behavior: solution evolved")
                    else:
                        print(f"⚠️  Burgers behavior: magnitude changed from {initial_max:.3f} to {final_max:.3f}")
                
                print(f"✅ {case['name']} with {opt['name']} completed successfully")
                
        except Exception as e:
            print(f"❌ Error in {case['name']} test: {e}")
            import traceback
            traceback.print_exc()

def test_convergence():
    """Test convergence behavior with different configurations"""
    print("\n=== Testing Convergence ===")
    
    # Simple advection equation for convergence testing
    eqn_str = "D(u, t) + D(u, x)"
    in_vars = "t, x"
    out_vars = "u"
    
    layers = [2, 20, 1]
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    
    # Test different network sizes
    network_configs = [
        {"layers": [2, 10, 1], "name": "Small"},
        {"layers": [2, 20, 1], "name": "Medium"},
        {"layers": [2, 40, 1], "name": "Large"}
    ]
    
    for config in network_configs:
        print(f"\nTesting {config['name']} network: {config['layers']}")
        
        try:
            tg = TrainingGround(config['layers'], lb, ub, "tanh", "Glorot Uniform", 
                              "Dirichlet", None, 500, "Regular", None, eqn_str, in_vars, out_vars, "Initial")
            
            # Simple training data
            X_i = np.array([[0.0, x] for x in np.linspace(0, 1, 20)])
            u_i = np.array([[np.sin(np.pi * x)] for x in np.linspace(0, 1, 20)])
            X_b = np.array([[t, 0.0] for t in np.linspace(0, 1, 15)] + [[t, 1.0] for t in np.linspace(0, 1, 15)])
            u_b = np.array([[0.0] for _ in range(30)])
            X_f = np.random.uniform(0, 1, (500, 2))
            
            train_data = {'X_i': X_i, 'u_i': u_i, 'X_b': X_b, 'u_b': u_b, 'X_f': X_f}
            
            # Train
            train_config = {'Optimizer': 'Adam', 'learning_rate': 0.01, 'Iterations': 150}
            training_time = tg.train(train_config, train_data)
            
            # Evaluate final loss
            final_loss = tg.loss_func(
                torch.tensor(X_i, dtype=torch.float32, requires_grad=True),
                torch.tensor(u_i, dtype=torch.float32),
                torch.tensor(X_b, dtype=torch.float32, requires_grad=True),
                torch.tensor(u_b, dtype=torch.float32),
                torch.tensor(X_f, dtype=torch.float32, requires_grad=True)
            )
            
            print(f"Final loss: {final_loss.item():.6f}")
            print(f"Training time: {training_time:.2f}s")
            
            # Test prediction accuracy at initial condition
            pred_i = tg.predict(X_i)
            mse = np.mean((pred_i - u_i)**2)
            print(f"Initial condition MSE: {mse:.6f}")
            
            if final_loss.item() < 0.5:
                print("✅ Good convergence")
            elif final_loss.item() < 1.0:
                print("⚠️  Moderate convergence")
            else:
                print("❌ Poor convergence")
                
        except Exception as e:
            print(f"❌ Error in convergence test: {e}")

if __name__ == "__main__":
    test_original_examples()
    test_convergence()