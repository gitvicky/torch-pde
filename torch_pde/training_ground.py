#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:17:47 2020

@author: Vicky

Neural PDE - PyTorch
Module : Model

Training Ground Class which houses all the associated training functions - loss functions, gradient functions, callbacks, training loops and evaluation functions 
"""
import time
import numpy as np
import torch

from .network import Network
from .pde import PDE
from . import boundary_conditions
from .sampler import Sampler
from . import options 
from . import qnw

class TrainingGround(Network, Sampler, PDE):
    
    def __init__(self, layers, lb, ub, activation, initializer, BC, BC_Vals, N_f, network_type, pde_func, eqn_str, in_vars, out_vars, sampler):
        print(f"DEBUG: TrainingGround.__init__ called with sampler={sampler}")
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
        Sampler.__init__(self, N_f, subspace_N = int(N_f/10))   #Percentage to be sampled from the subspace. 
        PDE.__init__(self, eqn_str, in_vars, out_vars)
        
        self.layers = layers 
        self.input_size = self.layers[0]
        self.output_size = self.layers[-1]
        
        self.bc = boundary_conditions.select(BC)
        
        if network_type == 'Regular':
            self.model = Network.initialize_NN(self)
        elif network_type == 'Resnet':
            self.model = Network.initialize_resnet(self, num_blocks=2)
        else:
            raise ValueError("Unknown Network Type. It should be either 'Regular' or 'Resnet'")
        
        self.trainable_params = list(self.model.parameters())
        
        self.pde = PDE.func  #Implicit  
        # self.pde = pde_func #Explicit
        
    def parameters(self):
        """Return the model parameters for compatibility with PyTorch optimizers and QN methods"""
        return self.model.parameters()
        
        self.loss_list =[]
        self.sampling_method = sampler
        print(f"DEBUG: sampler value = {sampler}")

    def ic_func(self, X, u):
        u_pred = self.model(X)
        ic_loss = u_pred - u
        return ic_loss
    
    def bc_func(self, X, u):
        bc_loss = self.bc(self.model, X, u)
        return bc_loss

    def pde_func(self, X):
        pde_loss = self.pde(self, X)
        return pde_loss

    def loss_func(self, X_i, u_i, X_b, u_b, X_f):
        initial_loss = self.ic_func(X_i, u_i)
        boundary_loss = self.bc_func(X_b, u_b)
        domain_loss = self.pde_func(X_f)
                
        return torch.mean(torch.square(initial_loss)) + \
                            torch.mean(torch.square(boundary_loss)) + \
                            torch.mean(torch.square(domain_loss))

    def loss_and_gradients(self, X_i, u_i, X_b, u_b, X_f):
        """
        
        Parameters
        ----------
        X_i : NUMPY ARRAY
            Initial input points.
        u_i : NUMPY ARRAY
            Initial outputs.
        X_b : NUMPY ARRAY
            Boundary input points.
        u_b : NUMPY ARRAY
            Boundary outputs.
        X_f : NUMPY ARRAY
            Domain input points.
        
        Returns
        -------
        model_loss : TENSOR
            Sum of Initial, Boundary and Domain MSE loss
        model_gradients : TENSOR
            Loss gradient with respect to the model trainable params. 
        
        """
        # Convert numpy arrays to torch tensors
        X_i_tensor = torch.tensor(X_i, dtype=torch.float32, requires_grad=True)
        u_i_tensor = torch.tensor(u_i, dtype=torch.float32)
        X_b_tensor = torch.tensor(X_b, dtype=torch.float32, requires_grad=True)
        u_b_tensor = torch.tensor(u_b, dtype=torch.float32)
        X_f_tensor = torch.tensor(X_f, dtype=torch.float32, requires_grad=True)
        
        model_loss = self.loss_func(X_i_tensor, u_i_tensor, X_b_tensor, u_b_tensor, X_f_tensor)
        
        # Compute gradients
        model_gradients = torch.autograd.grad(model_loss, self.trainable_params, create_graph=True)
        
        return model_loss, model_gradients

    def callback_GD(self, it, loss_value):
        elapsed = time.time() - self.init_time
        self.loss_list.append(loss_value.item())
        print('GD.  It: %d, Loss: %.3e, Time: %.2f' % 
                  (it, loss_value.item(), elapsed))
        self.init_time = time.time()
        
    def train(self, train_config, train_data):
        start_time = time.time()
        
        optimizer_class, lr, kind = options.get_optimizer(name=train_config['Optimizer'], lr=train_config['learning_rate'])
        nIter = train_config['Iterations']
         
        X_i = train_data['X_i']
        u_i = train_data['u_i']
        X_b = train_data['X_b']
        u_b = train_data['u_b']
        X_f = train_data['X_f']
                 
        self.init_time = time.time()
        
        if kind == "GD":
            if self.sampling_method == 'Initial':
                nIter_2 = nIter
            else : 
                nIter_2 = int(nIter/2)
              
            # Create optimizer with model parameters
            optimizer = optimizer_class(self.trainable_params, lr=lr)
             
            for it in range(nIter):
                model_loss, model_gradients = self.loss_and_gradients(X_i, u_i, X_b, u_b, X_f)
                 
                # Zero gradients
                optimizer.zero_grad()
                 
                # Update parameters using optimizer
                for param, grad in zip(self.trainable_params, model_gradients):
                    if grad is not None:
                        param.grad = grad
                 
                optimizer.step()
                 
                if it%10 == 0:
                    self.callback_GD(it, model_loss)
         
        elif kind == "QN_Scipy":
            from scipy.optimize import minimize
            func = qnw.Scipy_Keras_Wrapper(self.model, self.loss_func, X_i, u_i, X_b, u_b, X_f)
            # convert initial model parameters to a 1D tensor
            init_params = torch.cat([p.view(-1) for p in self.model.parameters()])
            init_params_np = init_params.detach().numpy()
             
            # Use L-BFGS for QN methods
            result = minimize(func, init_params_np, jac=True, method='L-BFGS-B')
             
            # Update model parameters
            new_params = result.x
            param_index = 0
            for param in self.model.parameters():
                param_size = param.numel()
                param.data = torch.tensor(new_params[param_index:param_index+param_size].reshape(param.shape), dtype=param.data.dtype)
                param_index += param_size
        
        elif kind == "QN_PyTorch":
            # Use PyTorch's L-BFGS optimizer
            optimizer, closure = qnw.PyTorch_LBFGS_Wrapper(self.model, self.loss_func, X_i, u_i, X_b, u_b, X_f)
            
            # Run L-BFGS optimization
            optimizer.step(closure)
            
            # Get final loss
            final_loss = closure()
            print(f'PyTorch L-BFGS Final Loss: {final_loss.item():.3e}')
        
        end_time = time.time() - start_time 
        return end_time

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return self.model(X_tensor).detach().numpy()
    
    def retrain(self, model, train_config, train_data):
        self.model = model
        self.trainable_params = list(self.model.parameters())
        
        return self.train(train_config, train_data)