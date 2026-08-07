#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 2026

@author: Vicky

Local example to test the torch-pde package and plot results
"""
import os
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt

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
print("Initializing model...")
model = torch_pde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)

# Train with Adam first
print("\n=== Training with Adam ===")
train_config_adam = {'Optimizer': 'adam',
                     'learning_rate': 0.001, 
                     'Iterations' : 10000}

start_time = time.time()
time_adam = model.train(train_config_adam, training_data)
adam_time = time.time() - start_time
print(f"Adam Training Time: {adam_time:.2f} seconds")

# Train with PyTorch L-BFGS
print("\n=== Training with PyTorch L-BFGS ===")
train_config_pytorch = {'Optimizer': 'L-BFGS-PyTorch',
                        'learning_rate': None, 
                        'Iterations' : 100}

start_time = time.time()
time_pytorch = model.train(train_config_pytorch, training_data)
pytorch_time = time.time() - start_time
print(f"PyTorch L-BFGS Training Time: {pytorch_time:.2f} seconds")

# Make predictions
u_pred = model.predict(X_star)
u_pred = np.reshape(u_pred, np.shape(Exact))

# Calculate error
error = np.abs(Exact - u_pred)
mse = np.mean(error**2)
print(f"\nFinal MSE: {mse:.3e}")

# Create plots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot exact solution
im1 = axes[0].imshow(Exact, extent=[x.min(), x.max(), t.max(), t.min()], aspect='auto')
axes[0].set_title('Exact Solution')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
fig.colorbar(im1, ax=axes[0])

# Plot predicted solution
im2 = axes[1].imshow(u_pred, extent=[x.min(), x.max(), t.max(), t.min()], aspect='auto')
axes[1].set_title('Predicted Solution (PyTorch L-BFGS)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
fig.colorbar(im2, ax=axes[1])

# Plot error
im3 = axes[2].imshow(error, extent=[x.min(), x.max(), t.max(), t.min()], aspect='auto')
axes[2].set_title('Absolute Error')
axes[2].set_xlabel('x')
axes[2].set_ylabel('t')
fig.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig('burgers_solution_comparison.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'burgers_solution_comparison.png'")

# Plot training times comparison
fig2, ax = plt.subplots(figsize=(8, 4))
training_methods = ['Adam', 'PyTorch L-BFGS']
training_times = [adam_time, pytorch_time]
bars = ax.bar(training_methods, training_times, color=['blue', 'orange'])
ax.set_title('Training Time Comparison')
ax.set_ylabel('Time (seconds)')
ax.set_xlabel('Optimization Method')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}s',
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
print("Training time comparison plot saved as 'training_time_comparison.png'")

print("\nExample completed successfully!")