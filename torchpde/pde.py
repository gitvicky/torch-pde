#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:58:15 2020

@author: Vicky

Neural PDE - PyTorch
Module : PDE Module 

Module with the PDE class that takes in the user defined pde parameters and creates a symbolic expression which 
is executed using lambdify. 
"""

import numpy as np
import torch
import torch.autograd as autograd
import sympy

from sympy.parsing.sympy_parser import parse_expr


class PDE(object):
    def __init__(self, eqn_str, in_vars, out_vars):
        """
        Parameters
        ----------
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
        self.num_inputs = len(in_vars)
        self.num_outputs = len(out_vars)
        
        in_vars = sympy.symbols(in_vars)
        out_vars = sympy.symbols(out_vars)
        
        if self.num_outputs == 1:
           self.all_vars = list(in_vars) + [out_vars]
        else: 
            self.all_vars = list(in_vars) + list(out_vars)

        self.expr = parse_expr(eqn_str)
    
        # Create a lambdified function with PyTorch operations
        self.fn = sympy.lambdify(self.all_vars, self.expr, 
                                [{'D': self.first_deriv, 
                                  'D2': self.second_deriv, 
                                  'D3': self.third_deriv}, 
                                 'numpy'])  # We'll handle torch tensors manually
        
        

    def first_deriv(self, u, wrt):
        """Compute first derivative using PyTorch autograd"""
        return autograd.grad(u, wrt, 
                           grad_outputs=torch.ones_like(u),
                           retain_graph=True,
                           create_graph=True)[0]
    

    def second_deriv(self, u, wrt):
        """Compute second derivative using PyTorch autograd"""
        u_deriv = autograd.grad(u, wrt, 
                               grad_outputs=torch.ones_like(u),
                               retain_graph=True,
                               create_graph=True)[0]
        u_deriv2 = autograd.grad(u_deriv, wrt, 
                                grad_outputs=torch.ones_like(u_deriv),
                                retain_graph=True,
                                create_graph=True)[0]
        return u_deriv2
    

    def third_deriv(self, u, wrt):
        """Compute third derivative using PyTorch autograd"""
        u_deriv = autograd.grad(u, wrt, 
                               grad_outputs=torch.ones_like(u),
                               retain_graph=True,
                               create_graph=True)[0]
        u_deriv2 = autograd.grad(u_deriv, wrt, 
                                grad_outputs=torch.ones_like(u_deriv),
                                retain_graph=True,
                                create_graph=True)[0]
        u_deriv3 = autograd.grad(u_deriv2, wrt, 
                                grad_outputs=torch.ones_like(u_deriv2),
                                retain_graph=True,
                                create_graph=True)[0]
        return u_deriv3
    
    def func(self, model, X):
        """
        Evaluate PDE residual
        X should be a torch tensor with requires_grad=True
        """
        # Ensure X requires gradients
        if not X.requires_grad:
            X = X.requires_grad_(True)
        
        # Split inputs
        t = X[:, 0:1]
        x = X[:, 1:2]
        
        # Ensure individual components require grad for autograd
        t = t.requires_grad_(True)
        x = x.requires_grad_(True)
        
        # Concatenate and forward through model
        inputs = torch.cat([t, x], dim=1)
        u = model(inputs)
        
        # Evaluate PDE using lambdified function
        # Convert torch operations result to tensor if needed
        result = self.fn(t, x, u)
        
        # Ensure result is a torch tensor
        if not torch.is_tensor(result):
            result = torch.tensor(result, dtype=torch.float64)
            
        return result
