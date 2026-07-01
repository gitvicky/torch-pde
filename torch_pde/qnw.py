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


def PyTorch_LBFGS_Wrapper(model, loss_func, X_i, u_i, X_b, u_b, X_f):
    """
    Wrapper for PyTorch's L-BFGS optimizer.
    
    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.
    loss_func : function
        The loss function to minimize.
    X_i : numpy.ndarray
        Initial condition input points.
    u_i : numpy.ndarray
        Initial condition output values.
    X_b : numpy.ndarray
        Boundary condition input points.
    u_b : numpy.ndarray
        Boundary condition output values.
    X_f : numpy.ndarray
        Domain input points.
    
    Returns
    -------
    optimizer : torch.optim.LBFGS
        Configured L-BFGS optimizer.
    """
    # Convert numpy arrays to torch tensors
    X_i_tensor = torch.tensor(X_i, dtype=torch.float32, requires_grad=True)
    u_i_tensor = torch.tensor(u_i, dtype=torch.float32)
    X_b_tensor = torch.tensor(X_b, dtype=torch.float32, requires_grad=True)
    u_b_tensor = torch.tensor(u_b, dtype=torch.float32)
    X_f_tensor = torch.tensor(X_f, dtype=torch.float32, requires_grad=True)
    
    # Define closure function for L-BFGS
    def closure():
        optimizer.zero_grad()
        loss = loss_func(X_i_tensor, u_i_tensor, X_b_tensor, u_b_tensor, X_f_tensor)
        loss.backward()
        return loss
    
    # Create L-BFGS optimizer
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)
    
    return optimizer, closure


def Scipy_Keras_Wrapper(model, loss_func, X_i, u_i, X_b, u_b, X_f):
    # obtain the shapes of all trainable parameters in the model
    shapes = [p.shape for p in model.parameters()]
    n_tensors = len(shapes)
    
    # prepare required information for parameter conversion
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices
    
    for i, shape in enumerate(shapes):
        n = int(torch.prod(torch.tensor(shape)))
        idx.append(torch.reshape(torch.arange(count, count+n, dtype=torch.int32), shape))
        part.extend([i]*n)
        count += n
    
    part = torch.tensor(part)
    
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tensor.
        Args:
            params_1d [in]: a 1D tensor representing the model's trainable parameters.
        """
        # Convert numpy array to tensor if needed
        if isinstance(params_1d, np.ndarray):
            params_1d = torch.tensor(params_1d, dtype=torch.float32)
        
        # Split parameters using the idx list
        params = []
        for i, indices in enumerate(idx):
            param_flat = params_1d[indices.flatten()]
            params.append(param_flat.reshape(shapes[i]))
        
        # Update model parameters - convert generator to list
        param_list = list(model.parameters())
        for i, param in enumerate(params):
            param_list[i].data = param
    
    def val_and_grads_1d(params_1d):
        """A function that can be used by scipy optimizer.
        Args:
           params_1d [in]: a 1D tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`
        """
        start_time = time.time()
        
        # update the parameters in the model
        assign_new_model_parameters(params_1d)
        
        # Convert numpy arrays to torch tensors
        X_i_tensor = torch.tensor(X_i, dtype=torch.float32, requires_grad=True)
        u_i_tensor = torch.tensor(u_i, dtype=torch.float32)
        X_b_tensor = torch.tensor(X_b, dtype=torch.float32, requires_grad=True)
        u_b_tensor = torch.tensor(u_b, dtype=torch.float32)
        X_f_tensor = torch.tensor(X_f, dtype=torch.float32, requires_grad=True)
        
        # calculate the loss
        loss_value = loss_func(X_i_tensor, u_i_tensor, X_b_tensor, u_b_tensor, X_f_tensor)
        
        # calculate gradients
        grads = torch.autograd.grad(loss_value, model.parameters(), create_graph=True)
        grads = torch.cat([g.view(-1) for g in grads])
        
        # print out iteration & loss
        val_and_grads_1d.iter += 1
        print('QN.  It: %d, Loss: %.3e, Time: %.2f' % 
                (val_and_grads_1d.iter, loss_value.item(),  np.round(time.time() - start_time, 3)))
        
        return loss_value.item(), grads.detach().numpy()
    
    # store these information as members so we can use them outside the scope
    val_and_grads_1d.iter = 0
    val_and_grads_1d.idx = idx
    val_and_grads_1d.part = part
    val_and_grads_1d.shapes = shapes
    val_and_grads_1d.assign_new_model_parameters = assign_new_model_parameters
    
    return val_and_grads_1d