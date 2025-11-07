#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:02:09 2020

Neural PDE - PyTorch
Module : Sampler

Samples points from the domain space according to the specified procedure. 
"""
import numpy as np
import torch
from pyDOE import lhs


def domain_sampler(N, lb, ub):
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    X_f = lb + (ub-lb)*lhs(2, N)
    return X_f


def boundary_sampler(N, lb, ub):
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    N_2 = int(N/2)
    X_b = lb + (ub-lb)*lhs(2, N)
    X_b[0:N_2, 1] = lb[1]
    X_b[N_2:2*N_2, 1] = ub[1]
    return X_b


def initial_sampler(N, lb, ub):
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    X_i = lb + (ub-lb)*lhs(2, N)
    X_i[:, 0] = np.zeros(N)
    return X_i


class Sampler(object):
    
    def __init__(self, N_samples, subspace_N):
        """
        Parameters
        ----------
        N_samples : INT
            Number of points sampled from the domain space. 
        subspace_N : INT
            Number of points we want to sample from each subspace. 

        Returns
        -------
        None.
        """
        self.N = N_samples
        self.ssN = subspace_N
        
    def residual(self, n, t_bounds):
        """Calculates the residual value for across each subspace"""
        residual_across_time = []
        
        for ii in range(n-1):
            lb_temp = np.asarray([t_bounds[ii], self.lb[1]])
            ub_temp = np.asarray([t_bounds[ii+1], self.ub[1]])
            X_f = lb_temp + (ub_temp-lb_temp)*lhs(self.input_size, self.ssN)
            
            # Convert to torch tensor for PDE evaluation
            X_f_tensor = torch.tensor(X_f, dtype=torch.float64, requires_grad=True)
            
            with torch.no_grad():
                pde_residual = self.pde_func(X_f_tensor)
                str_val = torch.mean(pde_residual**2).item()
            
            residual_across_time.append(str_val)

        return np.asarray(residual_across_time)

        
    def str_sampler(self):
        """Samples ssN points from the subspace with the worst residual performance"""
        n = int(self.N/self.ssN)
        t_bounds = np.linspace(self.lb[0], self.ub[0], n)
        
        residuals = self.residual(n, t_bounds)
        t_range_idx = np.argmax(residuals)
        print('\n')
        print("Time_Index : {}".format(t_range_idx))
        
        lb_temp = np.asarray([t_bounds[t_range_idx], self.lb[1]])
        ub_temp = np.asarray([t_bounds[t_range_idx+1], self.ub[1]])
        X_f = lb_temp + (ub_temp-lb_temp)*lhs(self.input_size, self.ssN) 
        
        return X_f
        
    
    def uniform_sampler(self):
        """Uniformly samples N points from across the domain space of interest"""
        X_f = self.lb + (self.ub-self.lb)*lhs(self.input_size, self.N) 
        return X_f
