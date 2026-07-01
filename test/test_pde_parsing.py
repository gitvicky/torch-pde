#!/usr/bin/env python3

import numpy as np
import torch
import sys
import os

# Add the torch_pde module to the path
sys.path.append('/Users/Vicky/Documents/UKAEA/Code/PINNs/torch-pde')

from torch_pde.network import Network
from torch_pde.pde import PDE

def test_advanced_pde_parsing():
    """Test advanced PDE parsing functionality"""
    print("Testing advanced PDE parsing...")
    
    # Test cases with more complex equations
    test_cases = [
        {
            "eqn_str": "D(u, t) + D(u, x)",
            "description": "Simple advection"
        },
        {
            "eqn_str": "D(u, t) + u*D(u, x)",
            "description": "Burgers equation"
        },
        {
            "eqn_str": "D(u, t) - D2(u, x)",
            "description": "Heat equation"
        },
        {
            "eqn_str": "D2(u, t) + D2(u, x) + u*D(u, x)",
            "description": "Complex equation with mixed derivatives"
        },
        {
            "eqn_str": "D(u, t) + D(u, x) + D(u, y) - D2(u, x) + u",
            "description": "3D equation with second derivatives"
        }
    ]
    
    for case in test_cases:
        print(f"\n--- Testing: {case['description']} ---")
        print(f"Equation: {case['eqn_str']}")
        
        try:
            # Create PDE
            pde = PDE(case['eqn_str'], "t, x, y", "u")
            
            print(f"Parsed terms: {pde.terms}")
            print(f"Has higher order derivatives: {pde.has_higher_order_derivatives}")
            print(f"Parsed structure: {pde.parsed_structure}")
            print(f"Derivative map: {pde.derivative_map}")
            
            # Create a simple model for testing
            layers = [3, 10, 1]  # 3 inputs (t, x, y)
            lb = np.array([0.0, 0.0, 0.0])
            ub = np.array([1.0, 1.0, 1.0])
            
            net = Network(layers, lb, ub, "tanh", "Glorot Uniform")
            model = net.initialize_NN()
            pde.model = model
            
            # Test evaluation
            X = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32, requires_grad=True)
            result = pde.func(X)
            
            print(f"Evaluation result shape: {result.shape}")
            print(f"Evaluation result: {result}")
            print("✅ Test passed")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

def test_expression_evaluation():
    """Test expression evaluation with different operators"""
    print("\n--- Testing Expression Evaluation ---")
    
    # Create a simple model
    layers = [2, 10, 1]
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    
    net = Network(layers, lb, ub, "tanh", "Glorot Uniform")
    model = net.initialize_NN()
    
    # Test different expressions
    expressions = [
        "D(u, t) + D(u, x)",
        "D(u, t) - D(u, x)",
        "D(u, t) * D(u, x)",
        "D(u, t) + u*D(u, x)",
        "D2(u, x) + D(u, t)",
        "D(u, t) + D(u, x) + u"
    ]
    
    X = torch.tensor([[0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    
    for expr in expressions:
        print(f"\nTesting expression: {expr}")
        try:
            pde = PDE(expr, "t, x", "u")
            pde.model = model
            
            result = pde.func(X)
            print(f"Result: {result}")
            print("✅ Evaluation successful")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    test_advanced_pde_parsing()
    test_expression_evaluation()