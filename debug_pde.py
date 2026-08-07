#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to investigate PDE computation issue
"""
import os
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt
import torch
import torch_pde

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Neural Network Hyperparameters
NN_parameters = {'Network_Type': 'Regular',
                'input_neurons' : 2,
                'output_neurons' : 1,
                'num_layers' : 4,
                'num_neurons' : 64,
                }

#Neural PDE Hyperparameters
NPDE_parameters = {'Sampling_Method': 'Initial',
                   'N_initial' : 100, #Number of Randomly sampled Data points from the IC vector
                   'N_boundary' : 100, #Number of Boundary Points
                   'N_domain' : 5000 #Number of Domain points generated
                  }

#PDE 
PDE_parameters = {'Inputs': 't, x',
                  'Outputs': 'u',
                  'Equation': 'D(u, t) + u*D(u, x) - 0.1*D2(u, x)',
                  'lower_range': [0.0, -8.0], #Float 
                  'upper_range': [10.0, 8.0], #Float
                  'Boundary_Condition': "Dirichlet", #Periodic 
                  'Boundary_Vals' : None,
                  'Initial_Condition': lambda x: -np.sin((np.pi*x)/8),
                  'Initial_Vals': None
                 }

# Data Location
data_loc = os.path.abspath('.') + '/Data/'
data = scipy.io.loadmat(data_loc +'burgers.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None])) 
u_star = Exact.flatten()[:,None]              

# Domain bounds
lb = X_star.min(0) 
ub = X_star.max(0)
      
X_i = np.hstack((T[0:1,:].T, X[0:1,:].T))
u_i = Exact[0:1,:].T

X_lb = np.hstack((T[:,0:1], X[:,0:1])) 
u_lb = Exact[:,0:1] 
X_ub = np.hstack((T[:,-1:], X[:,-1:])) 
u_ub = Exact[:,-1:] 

u_lb = np.zeros((len(u_lb),1))
u_ub = np.zeros((len(u_ub),1))  

X_b = np.vstack((X_lb, X_ub))
u_b = np.vstack((u_lb, u_ub))

X_f = torch_pde.sampler.domain_sampler(NPDE_parameters['N_domain'], lb, ub) 

N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']

idx = np.random.choice(X_i.shape[0], N_i, replace=False)
X_i = X_i[idx, :]
u_i = u_i[idx,:]

idx = np.random.choice(X_b.shape[0], N_b, replace=False)
X_b = X_b[idx, :] 
u_b = u_b[idx,:]


training_data = {'X_i': X_i, 'u_i': u_i,
                'X_b': X_b, 'u_b': u_b,
                'X_f': X_f}

# Initialize model
print("Initializing model...")
model = torch_pde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)

# Debug: Check PDE computation
print("\n=== Debug: PDE Computation ===")
X_f_tensor = torch.tensor(X_f, dtype=torch.float32, requires_grad=True)

print("Testing PDE function directly...")
try:
    pde_result = model.pde_func(X_f_tensor)
    print(f"PDE result shape: {pde_result.shape}")
    print(f"PDE result mean: {pde_result.mean().item():.3e}")
    print(f"PDE result std: {pde_result.std().item():.3e}")
    print(f"PDE result requires_grad: {pde_result.requires_grad}")
    
    # Check if PDE result has any non-zero values
    print(f"PDE result non-zero count: {torch.count_nonzero(pde_result).item()}")
    print(f"PDE result sample values: {pde_result[:5]}")
    
except Exception as e:
    print(f"Error in PDE computation: {e}")
    import traceback
    traceback.print_exc()

# Debug: Check model prediction
print("\n=== Debug: Model Prediction ===")
with torch.no_grad():
    pred_f = model.model(X_f_tensor)
    print(f"Model prediction shape: {pred_f.shape}")
    print(f"Model prediction mean: {pred_f.mean().item():.3e}")
    print(f"Model prediction std: {pred_f.std().item():.3e}")
    print(f"Model prediction sample values: {pred_f[:5]}")

# Debug: Check derivatives computation
print("\n=== Debug: Derivatives Computation ===")
try:
    # Test first derivative
    u = model.model(X_f_tensor)
    print(f"u shape: {u.shape}, requires_grad: {u.requires_grad}")
    
    # Test if autograd is working
    du_dt = torch.autograd.grad(u, X_f_tensor[:, 0:1], grad_outputs=torch.ones_like(u), create_graph=True)[0]
    print(f"du_dt shape: {du_dt.shape}, requires_grad: {du_dt.requires_grad}")
    print(f"du_dt mean: {du_dt.mean().item():.3e}, std: {du_dt.std().item():.3e}")
    
    # Test second derivative
    du_x = torch.autograd.grad(u, X_f_tensor[:, 1:2], grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_x2 = torch.autograd.grad(du_x, X_f_tensor[:, 1:2], grad_outputs=torch.ones_like(du_x), create_graph=True)[0]
    print(f"d2u_x2 shape: {d2u_x2.shape}, requires_grad: {d2u_x2.requires_grad}")
    print(f"d2u_x2 mean: {d2u_x2.mean().item():.3e}, std: {d2u_x2.std().item():.3e}")
    
    # Test Burgers equation manually
    burgers_manual = du_dt + u * du_x - 0.1 * d2u_x2
    print(f"Manual Burgers result mean: {burgers_manual.mean().item():.3e}")
    print(f"Manual Burgers result std: {burgers_manual.std().item():.3e}")
    
except Exception as e:
    print(f"Error in derivatives computation: {e}")
    import traceback
    traceback.print_exc()

print("\nDebug completed!")