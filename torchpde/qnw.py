#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:34:39 2020

@author: Vicky

Neural PDE - PyTorch

Module: Quasi-Newtonian Wrappers 

Help convert the Model variables 1D and back to the Model structure. 
"""
import torch
import numpy as np
import time 


class Scipy_Wrapper:
    """
    Wrapper to use scipy optimizers with PyTorch models
    """
    def __init__(self, model, loss_func, X_i, u_i, X_b, u_b, X_f, device):
        self.model = model
        self.loss_func = loss_func
        self.X_i = X_i
        self.u_i = u_i
        self.X_b = X_b
        self.u_b = u_b
        self.X_f = X_f
        self.device = device
        self.iter = 0
        
        # Get shapes of parameters
        self.shapes = []
        self.n_params = 0
        for p in self.model.parameters():
            self.shapes.append(p.shape)
            self.n_params += p.numel()
    
    def get_params(self):
        """Get model parameters as a flat numpy array"""
        params = []
        for p in self.model.parameters():
            params.append(p.data.cpu().numpy().ravel())
        return np.concatenate(params)
    
    def set_params(self, params_1d):
        """Set model parameters from a flat numpy array"""
        idx = 0
        for p in self.model.parameters():
            n_params = p.numel()
            p.data = torch.from_numpy(params_1d[idx:idx + n_params]).reshape(p.shape).to(self.device)
            idx += n_params
    
    def __call__(self, params_1d):
        """
        Compute loss and gradients for scipy optimizer
        
        Returns:
            loss_value (float): The loss value
            grads (numpy array): The gradients as a flat numpy array
        """
        start_time = time.time()
        
        # Set parameters
        self.set_params(params_1d)
        
        # Enable gradients
        for p in self.model.parameters():
            p.requires_grad_(True)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Compute loss
        loss = self.loss_func(self.X_i, self.u_i, self.X_b, self.u_b, self.X_f)
        
        # Compute gradients
        loss.backward()
        
        # Extract gradients
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.cpu().numpy().ravel())
            else:
                grads.append(np.zeros(p.numel()))
        grads = np.concatenate(grads)
        
        # Print progress
        self.iter += 1
        print('Scipy. It: %d, Loss: %.3e, Time: %.2f' % 
              (self.iter, loss.item(), time.time() - start_time))
        
        return loss.item(), grads


class LBFGS_Wrapper:
    """
    Wrapper for PyTorch's LBFGS optimizer to work similarly to TensorFlow Probability's version
    """
    def __init__(self, model, loss_func, X_i, u_i, X_b, u_b, X_f, device, 
                 max_iter=20, max_eval=25, tolerance_grad=1e-5, tolerance_change=1e-9):
        self.model = model
        self.loss_func = loss_func
        self.X_i = X_i
        self.u_i = u_i
        self.X_b = X_b
        self.u_b = u_b
        self.X_f = X_f
        self.device = device
        self.iter = 0
        
        # Create LBFGS optimizer
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=50,
            line_search_fn="strong_wolfe"
        )
    
    def closure(self):
        """Closure function for LBFGS"""
        if torch.is_grad_enabled():
            self.optimizer.zero_grad()
        
        # Compute loss
        loss = self.loss_func(self.X_i, self.u_i, self.X_b, self.u_b, self.X_f)
        
        if loss.requires_grad:
            loss.backward()
        
        # Print progress
        self.iter += 1
        if self.iter % 10 == 0:
            print('LBFGS. It: %d, Loss: %.3e' % (self.iter, loss.item()))
        
        return loss
    
    def optimize(self, n_steps=10):
        """Run optimization for n_steps"""
        for _ in range(n_steps):
            self.optimizer.step(self.closure)
