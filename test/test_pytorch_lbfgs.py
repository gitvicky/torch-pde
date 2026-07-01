#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 2026

@author: Vicky

Test script to compare SciPy L-BFGS and PyTorch L-BFGS optimizers
"""
import os
import numpy as np
import scipy.io
import time

import torch_pde

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
model = torch_pde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)

# Test SciPy L-BFGS
print("\n=== Testing SciPy L-BFGS ===")
train_config_scipy = {'Optimizer': 'L-BFGS',
                      'learning_rate': None, 
                      'Iterations' : None}

start_time = time.time()
time_scipy = model.train(train_config_scipy, training_data)
scipy_time = time.time() - start_time
print(f"SciPy L-BFGS Training Time: {scipy_time:.2f} seconds")

# Save SciPy model
model_scipy = model

# Reinitialize model for PyTorch test
model = torch_pde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)

# Test PyTorch L-BFGS
print("\n=== Testing PyTorch L-BFGS ===")
train_config_pytorch = {'Optimizer': 'L-BFGS-PyTorch',
                        'learning_rate': None, 
                        'Iterations' : None}

start_time = time.time()
time_pytorch = model.train(train_config_pytorch, training_data)
pytorch_time = time.time() - start_time
print(f"PyTorch L-BFGS Training Time: {pytorch_time:.2f} seconds")

# Save PyTorch model
model_pytorch = model

# Compare results
print("\n=== Comparison ===")
print(f"SciPy L-BFGS Time: {scipy_time:.2f} seconds")
print(f"PyTorch L-BFGS Time: {pytorch_time:.2f} seconds")
print(f"Time Difference: {abs(scipy_time - pytorch_time):.2f} seconds")

# Evaluate both models
u_pred_scipy = model_scipy.predict(X_star)
u_pred_pytorch = model_pytorch.predict(X_star)

# Calculate MSE
mse_scipy = np.mean((u_pred_scipy - u_star)**2)
mse_pytorch = np.mean((u_pred_pytorch - u_star)**2)

print(f"\nSciPy L-BFGS MSE: {mse_scipy:.3e}")
print(f"PyTorch L-BFGS MSE: {mse_pytorch:.3e}")

# Determine which performed better
if mse_scipy < mse_pytorch:
    print("SciPy L-BFGS performed better in terms of MSE")
    print(f"MSE difference: {mse_pytorch - mse_scipy:.3e}")
else:
    print("PyTorch L-BFGS performed better in terms of MSE")
    print(f"MSE difference: {mse_scipy - mse_pytorch:.3e}")