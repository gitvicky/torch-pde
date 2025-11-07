#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:37:25 2020

@author: Vicky

Neural PDE - PyTorch
Testing with Korteweg de Vries Equation

PDE: u_t + u*u_x + 0.0025*u_xxx 
IC: u(0, x) = cos(pi.x),
BC: Periodic 
Domain: t ∈ [0,1],  x ∈ [-1,1]
"""
# %% 
import os
import sys 
import numpy as np 
import scipy.io
import torch

sys.path.append("..")
import torchpde 

# %%
#Neural Network Hyperparameters
NN_parameters = {'Network_Type': 'Regular',
                'input_neurons' : 2,
                'output_neurons' : 1,
                'num_layers' : 4,
                'num_neurons' : 64
                }


#Neural PDE Hyperparameters
NPDE_parameters = {'Sampling_Method': 'Initial',
                   'N_initial' : 300, #Number of Randomly sampled Data points from the IC vector
                   'N_boundary' : 300, #Number of Boundary Points
                   'N_domain' : 20000 #Number of Domain points generated
                  }


#PDE 
PDE_parameters = {'Inputs': 't, x',
                  'Outputs': 'u',
                  'Equation': 'D(u, t) + u*D(u, x) + 0.0025*D3(u, x)',
                  'lower_range': [0.0, -1.0], #Float 
                  'upper_range': [1.0, 1.0], #Float
                  'Boundary_Condition': "Periodic",
                  'Boundary_Vals' : None,
                  'Initial_Condition': lambda x: np.cos(np.pi*x),
                  'Initial_Vals': None
                 }


# %%

#Using Simulation Data at the Initial and Boundary Values (BC would be Dirichlet under that case)

N_f = NPDE_parameters['N_domain']
N_i = NPDE_parameters['N_initial']
N_b = NPDE_parameters['N_boundary']

# Data Location
data_loc = os.path.abspath('..') + '/Data/'
data = scipy.io.loadmat(data_loc + 'KdV.mat')

t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['uu']).T

X, T = np.meshgrid(x, t)

X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None])) #Flattened array with the inputs X and T 
u_star = Exact.flatten()[:, None]              

# Domain bounds
lb = X_star.min(0) #Lower bounds of x and t 
ub = X_star.max(0) #Upper bounds of x and t
    
X_i = np.hstack((T[0:1, :].T, X[0:1, :].T)) #Initial condition value of X (x=-1....1) and T (t = 0) 
u_i = Exact[0:1, :].T #Initial Condition value of the field u

X_lb = np.hstack((T[:, 0:1], X[:, 0:1])) #Lower Boundary condition value of X (x = -1) and T (t = 0...0.99)
u_lb = Exact[:, 0:1] #Bound Condition value of the field u at (x = -1) and T (t = 0...0.99)
X_ub = np.hstack((T[:, -1:], X[:, -1:])) #Upper Boundary condition value of X (x = 1) and T (t = 0...0.99)
u_ub = Exact[:, -1:] #Bound Condition value of the field u at (x = 1) and T (t = 0...0.99)

X_b = np.vstack((X_lb, X_ub))
u_b = np.vstack((u_lb, u_ub))

X_f = torchpde.sampler.domain_sampler(N_f, lb, ub)

idx = np.random.choice(X_i.shape[0], N_i, replace=False)
X_i = X_i[idx, :] #Randomly Extract the N_u number of x and t values. 
u_i = u_i[idx, :] #Extract the N_u number of field values 

idx = np.random.choice(X_b.shape[0], N_b, replace=False)
X_b = X_b[idx, :] #Randomly Extract the N_u number of x and t values. 
u_b = u_b[idx, :] #Extract the N_u number of field values 


training_data = {'X_i': X_i, 'u_i': u_i,
                'X_b': X_b, 'u_b': u_b,
                'X_f': X_f}

# %%

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%

model = torchpde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)

# %%
# Train with Adam optimizer
train_config = {'Optimizer': 'adam',
                 'learning_rate': 0.001, 
                 'Iterations' : 5000}

time_GD = model.train(train_config, training_data)
print(f"Adam training time: {time_GD:.2f} seconds")

# %%
# Train with L-BFGS optimizer
train_config = {'Optimizer': 'L-BFGS',
                 'learning_rate': 1.0, 
                 'Iterations' : 500}

time_QN = model.train(train_config, training_data)
print(f"L-BFGS training time: {time_QN:.2f} seconds")

# %%
# Evaluation
data_loc = os.path.abspath('..') + '/Data/'
data = scipy.io.loadmat(data_loc + 'KdV.mat')

t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['uu']).T

X, T = np.meshgrid(x, t)

X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None])) #Flattened array with the inputs X and T 
u_star = Exact.flatten()[:, None]              

u_pred = model.predict(X_star)
u_pred = np.reshape(u_pred, np.shape(Exact))

# Calculate error
error = np.linalg.norm(u_star - u_pred.flatten()[:, None], 2) / np.linalg.norm(u_star, 2)
print(f"Relative L2 error: {error:.3e}")

# Visualize results
torchpde.plotter.evolution_plot(Exact, u_pred)
