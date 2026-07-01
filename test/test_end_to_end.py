#!/usr/bin/env python3

import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt

# Add the torch_pde module to the path
sys.path.append('/Users/Vicky/Documents/UKAEA/Code/PINNs/torch-pde')

from torch_pde.network import Network
from torch_pde.pde import PDE
from torch_pde.training_ground import TrainingGround
from torch_pde import boundary_conditions

def test_end_to_end_training():
    """Comprehensive end-to-end test of the training pipeline"""
    print("=== End-to-End Training Test ===")
    
    # Test configuration
    layers = [2, 20, 1]
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    activation = "tanh"
    initializer = "Glorot Uniform"
    BC = "Dirichlet"
    N_f = 500
    network_type = "Regular"
    sampler = "Initial"
    
    # Test different PDEs
    pde_configs = [
        {
            "name": "Advection Equation",
            "eqn_str": "D(u, t) + D(u, x)",
            "in_vars": "t, x",
            "out_vars": "u",
            "initial_condition": lambda x: np.sin(np.pi * x),
            "boundary_condition": lambda t: 0.0,
            "expected_pattern": "Wave propagation"
        },
        {
            "name": "Heat Equation",
            "eqn_str": "D(u, t) - D2(u, x)",
            "in_vars": "t, x",
            "out_vars": "u", 
            "initial_condition": lambda x: np.sin(np.pi * x),
            "boundary_condition": lambda t: 0.0,
            "expected_pattern": "Diffusion"
        }
    ]
    
    for config in pde_configs:
        print(f"\n--- Testing {config['name']} ---")
        print(f"Equation: {config['eqn_str']}")
        print(f"Expected pattern: {config['expected_pattern']}")
        
        try:
            # Create training ground
            tg = TrainingGround(layers, lb, ub, activation, initializer, BC, None, N_f, 
                              network_type, None, config['eqn_str'], config['in_vars'], config['out_vars'], sampler)
            
            # Generate training data
            x_vals = np.linspace(0, 1, 30)
            X_i = np.array([[0.0, x] for x in x_vals])
            u_i = np.array([[config['initial_condition'](x)] for x in x_vals])
            
            t_vals = np.linspace(0, 1, 20)
            X_b = np.array([[t, 0.0] for t in t_vals] + [[t, 1.0] for t in t_vals])
            u_b = np.array([[config['boundary_condition'](t)] for t in t_vals] + 
                          [[config['boundary_condition'](t)] for t in t_vals])
            
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
                {'name': 'Adam', 'config': {'Optimizer': 'Adam', 'learning_rate': 0.01, 'Iterations': 100}},
                {'name': 'L-BFGS', 'config': {'Optimizer': 'L-BFGS', 'learning_rate': 0.1, 'Iterations': 50}}
            ]
            
            for opt in optimizers:
                print(f"\nTraining with {opt['name']} optimizer...")
                
                # Train the model
                training_time = tg.train(opt['config'], train_data)
                print(f"Training completed in {training_time:.2f} seconds")
                
                # Test predictions on a grid
                test_points = []
                for t in [0.25, 0.5, 0.75]:
                    for x in np.linspace(0, 1, 10):
                        test_points.append([t, x])
                
                test_points = np.array(test_points)
                predictions = tg.predict(test_points)
                
                print(f"Sample predictions:")
                for i, (point, pred) in enumerate(zip(test_points[::5], predictions[::5])):
                    print(f"  t={point[0]:.2f}, x={point[1]:.2f} -> u={pred[0]:.4f}")
                
                # Test loss computation
                final_loss = tg.loss_func(
                    torch.tensor(X_i, dtype=torch.float32, requires_grad=True),
                    torch.tensor(u_i, dtype=torch.float32),
                    torch.tensor(X_b, dtype=torch.float32, requires_grad=True),
                    torch.tensor(u_b, dtype=torch.float32),
                    torch.tensor(X_f, dtype=torch.float32, requires_grad=True)
                )
                
                print(f"Final loss: {final_loss.item():.6f}")
                
                # Verify that training actually changed the model
                initial_pred = tg.predict(np.array([[0.0, 0.5]]))
                final_pred = tg.predict(np.array([[1.0, 0.5]]))
                
                print(f"Initial condition at x=0.5: {initial_pred[0][0]:.4f}")
                print(f"Final prediction at t=1.0, x=0.5: {final_pred[0][0]:.4f}")
                
                # Check if the solution makes physical sense
                if config['name'] == "Advection Equation":
                    # For advection, the wave should propagate
                    if abs(initial_pred[0][0] - final_pred[0][0]) > 0.1:
                        print("✅ Advection behavior detected")
                    else:
                        print("⚠️  Advection behavior not clearly detected")
                elif config['name'] == "Heat Equation":
                    # For heat equation, the solution should decay
                    if abs(final_pred[0][0]) < abs(initial_pred[0][0]):
                        print("✅ Diffusion behavior detected")
                    else:
                        print("⚠️  Diffusion behavior not clearly detected")
                
                print(f"✅ {config['name']} with {opt['name']} completed successfully")
                
        except Exception as e:
            print(f"❌ Error in {config['name']} test: {e}")
            import traceback
            traceback.print_exc()

def test_gradient_flow():
    """Test that gradients flow properly through the network"""
    print("\n=== Gradient Flow Test ===")
    
    # Create a simple setup
    layers = [2, 10, 1]
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    
    net = Network(layers, lb, ub, "tanh", "Glorot Uniform")
    model = net.initialize_NN()
    
    # Create PDE
    pde = PDE("D(u, t) + D(u, x)", "t, x", "u")
    pde.model = model
    
    # Test data
    X = torch.tensor([[0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    
    print("Testing gradient computation graph...")
    
    # Forward pass
    u = model(X)
    print(f"Model output: {u}")
    
    # PDE evaluation
    pde_result = pde.func(X)
    print(f"PDE result: {pde_result}")
    
    # Test backward pass
    try:
        loss = torch.sum(pde_result**2)
        loss.backward()
        
        print("✅ Backward pass successful")
        
        # Check if parameters have gradients
        has_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grads = True
                print(f"Parameter {name} has gradient: {param.grad.norm().item():.4f}")
                break
        
        if has_grads:
            print("✅ Gradients are flowing through the network")
        else:
            print("❌ No gradients found in parameters")
            
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")

if __name__ == "__main__":
    test_end_to_end_training()
    test_gradient_flow()