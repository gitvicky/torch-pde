#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:17:47 2020

@author: Vicky

Neural PDE - PyTorch
Module : Training Ground

Training Ground Class which houses all the associated training functions - loss functions, gradient functions, callbacks, training loops and evaluation functions 
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from .network import Network
from .pde import PDE
from . import boundary_conditions
from .sampler import Sampler
from . import options 
from . import qnw


class TrainingGround(Network, Sampler, PDE):
    
    def __init__(self, layers, lb, ub, activation, initializer, BC, BC_Vals, N_f, network_type, pde_func, eqn_str, in_vars, out_vars, sampler):
        """
        Parameters
        ----------
        layers : LIST
            Number of neurons in each layer
        lb : ARRAY
            Lower Range of the time and space domain
        ub : ARRAY
            Upper Range of the time and space domain
        activation : STR
            Name of the activation Function
        initializer : STR
            Name of the Initialiser for the neural network weights
        N_f : INT
            Number of points sampled from the domain space.
        pde_func : FUNC
            Explicitly defined domain function.
        eqn_str : STR
            The PDE in string with the specified format.
        in_vars : INT
            Number of input variables.
        out_vars : INT
            Number of output variables.

        Returns
        -------
        None.
        """
        
        Network.__init__(self, layers, lb, ub, activation, initializer)
        Sampler.__init__(self, N_f, subspace_N=int(N_f/10))
        PDE.__init__(self, eqn_str, in_vars, out_vars)
        
        self.layers = layers 
        self.input_size = self.layers[0]
        self.output_size = self.layers[-1]
        
        # Convert lb and ub to numpy for sampler compatibility
        self.lb = np.asarray(lb) if not isinstance(lb, np.ndarray) else lb
        self.ub = np.asarray(ub) if not isinstance(ub, np.ndarray) else ub
        
        self.bc = boundary_conditions.select(BC)
        
        if network_type == 'Regular':
            self.model = Network.initialize_NN(self)
        elif network_type == 'Resnet':
            self.model = Network.initialize_resnet(self, num_blocks=2)
        else:
            raise ValueError("Unknown Network Type. It should be either 'Regular' or 'Resnet'")

        # Move model to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.double()  # Ensure double precision
        
        self.pde = PDE.func  # Implicit 
        
        self.loss_list = []
        self.sampler = sampler
        
        
    def ic_func(self, X, u):
        """Initial condition loss"""
        u_pred = self.model(X)
        ic_loss = u_pred - u
        return ic_loss

    
    def bc_func(self, X, u):
        """Boundary condition loss"""
        bc_loss = self.bc(self.model, X, u)
        return bc_loss
     
    
    def pde_func(self, X):
        """PDE residual loss"""
        # Ensure X requires gradients
        if not X.requires_grad:
            X = X.requires_grad_(True)
        pde_loss = self.pde(self, X)
        return pde_loss
    
    
    def loss_func(self, X_i, u_i, X_b, u_b, X_f):
        """Combined loss function"""
        initial_loss = self.ic_func(X_i, u_i)
        boundary_loss = self.bc_func(X_b, u_b)
        domain_loss = self.pde_func(X_f)
        
        loss_i = torch.mean(initial_loss**2)
        loss_b = torch.mean(boundary_loss**2)
        loss_f = torch.mean(domain_loss**2)
        
        return loss_i + loss_b + loss_f
    
    
    def callback_GD(self, it, loss_value):
        """Callback for gradient descent"""
        elapsed = time.time() - self.init_time
        self.loss_list.append(loss_value)
        print('GD.  It: %d, Loss: %.3e, Time: %.2f' % 
                  (it, loss_value, elapsed))
        self.init_time = time.time()
        
    
    def train(self, train_config, train_data):
        """Main training function"""
        start_time = time.time()
        
        optimizer_name = train_config['Optimizer']
        lr = train_config['learning_rate']
        nIter = train_config['Iterations']
        
        # Convert training data to torch tensors
        X_i = torch.tensor(train_data['X_i'], dtype=torch.float64, requires_grad=True).to(self.device)
        u_i = torch.tensor(train_data['u_i'], dtype=torch.float64).to(self.device)
        X_b = torch.tensor(train_data['X_b'], dtype=torch.float64, requires_grad=True).to(self.device)
        u_b = torch.tensor(train_data['u_b'], dtype=torch.float64).to(self.device)
        X_f = torch.tensor(train_data['X_f'], dtype=torch.float64, requires_grad=True).to(self.device)
        
        self.init_time = time.time()
        
        if optimizer_name in ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "adamw"]:
            # Standard gradient descent optimizers
            optimizer_class, _ = options.get_optimizer(optimizer_name)
            
            if optimizer_name == "adadelta":
                optimizer = optimizer_class(self.model.parameters())
            else:
                optimizer = optimizer_class(self.model.parameters(), lr=lr)
            
            self.model.train()
            
            for it in range(nIter):
                optimizer.zero_grad()
                
                # Forward pass
                loss = self.loss_func(X_i, u_i, X_b, u_b, X_f)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                if it % 10 == 0:
                    self.callback_GD(it, loss.item())
                    
        elif optimizer_name in ["LBFGS", "L-BFGS"]:
            # L-BFGS optimizer
            optimizer = optim.LBFGS(self.model.parameters(), 
                                  lr=lr if lr else 1.0,
                                  max_iter=20,
                                  max_eval=25,
                                  history_size=50,
                                  tolerance_grad=1e-5,
                                  tolerance_change=1e-9,
                                  line_search_fn="strong_wolfe")
            
            self.model.train()
            iter_count = [0]
            
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                loss = self.loss_func(X_i, u_i, X_b, u_b, X_f)
                if loss.requires_grad:
                    loss.backward()
                
                # Print progress
                if iter_count[0] % 10 == 0:
                    print('LBFGS. It: %d, Loss: %.3e' % (iter_count[0], loss.item()))
                iter_count[0] += 1
                
                return loss
            
            # Run optimization for specified iterations
            for _ in range(nIter // 20 if nIter else 10):  # Divide by max_iter per step
                optimizer.step(closure)
                
        elif optimizer_name in ["L-BFGS-B", "BFGS", "Nelder-Mead", "Powell", "CG", "TNC", "COBYLA", "SLSQP"]:
            # Scipy optimizers
            func = qnw.Scipy_Wrapper(self.model, self.loss_func, X_i, u_i, X_b, u_b, X_f, self.device)
            
            # Get initial parameters
            init_params = func.get_params()
            
            # Run optimization
            from scipy import optimize
            result = optimize.minimize(fun=func,
                                      x0=init_params,
                                      jac=True,
                                      method=optimizer_name,
                                      options={'maxiter': nIter if nIter else 500,
                                              'disp': True})
            
            # Set optimized parameters back to model
            func.set_params(result.x)
            
        end_time = time.time() - start_time 
        return end_time
        
        
    def predict(self, X):
        """Make predictions with the trained model"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float64).to(self.device)
            # Normalize input
            X_norm = 2.0 * (X_tensor - torch.tensor(self.lb, dtype=torch.float64).to(self.device)) / \
                    (torch.tensor(self.ub, dtype=torch.float64).to(self.device) - 
                     torch.tensor(self.lb, dtype=torch.float64).to(self.device)) - 1.0
            predictions = self.model(X_norm)
            return predictions.cpu().numpy()
    
    
    def retrain(self, model, train_config, train_data):
        """Retrain with a given model"""
        self.model = model
        self.model = self.model.to(self.device)
        
        return self.train(train_config, train_data)
