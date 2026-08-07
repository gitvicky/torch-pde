#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to identify training issues
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

# Debug: Check initial loss
print("\n=== Debug: Initial Loss Check ===")
X_i_tensor = torch.tensor(X_i, dtype=torch.float32, requires_grad=True)
u_i_tensor = torch.tensor(u_i, dtype=torch.float32)
X_b_tensor = torch.tensor(X_b, dtype=torch.float32, requires_grad=True)
u_b_tensor = torch.tensor(u_b, dtype=torch.float32)
X_f_tensor = torch.tensor(X_f, dtype=torch.float32, requires_grad=True)

initial_loss = model.loss_func(X_i_tensor, u_i_tensor, X_b_tensor, u_b_tensor, X_f_tensor)
print(f"Initial Loss: {initial_loss.item():.3e}")

# Debug: Check gradients
print("\n=== Debug: Gradient Check ===")
model_gradients = torch.autograd.grad(initial_loss, model.trainable_params, create_graph=True)
print(f"Number of gradients: {len(model_gradients)}")
for i, grad in enumerate(model_gradients):
    if grad is not None:
        print(f"Gradient {i} norm: {grad.norm().item():.3e}")
    else:
        print(f"Gradient {i}: None")

# Debug: Check individual loss components
print("\n=== Debug: Loss Components ===")
initial_loss_component = model.ic_func(X_i_tensor, u_i_tensor)
boundary_loss_component = model.bc_func(X_b_tensor, u_b_tensor)
domain_loss_component = model.pde_func(X_f_tensor)

print(f"Initial Loss Component: {torch.mean(torch.square(initial_loss_component)).item():.3e}")
print(f"Boundary Loss Component: {torch.mean(torch.square(boundary_loss_component)).item():.3e}")
print(f"Domain Loss Component: {torch.mean(torch.square(domain_loss_component)).item():.3e}")

# Debug: Check model predictions
print("\n=== Debug: Model Predictions ===")
with torch.no_grad():
    # Check predictions on initial conditions
    pred_i = model.model(torch.tensor(X_i, dtype=torch.float32))
    print(f"Initial condition prediction MSE: {torch.mean((pred_i - torch.tensor(u_i, dtype=torch.float32))**2).item():.3e}")
    
    # Check predictions on boundary conditions
    pred_b = model.model(torch.tensor(X_b, dtype=torch.float32))
    print(f"Boundary condition prediction MSE: {torch.mean((pred_b - torch.tensor(u_b, dtype=torch.float32))**2).item():.3e}")
    
    # Check predictions on domain points
    pred_f = model.model(torch.tensor(X_f, dtype=torch.float32))
    print(f"Domain prediction mean: {pred_f.mean().item():.3e}, std: {pred_f.std().item():.3e}")

print("\nDebug completed!")