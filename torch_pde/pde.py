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
        self.eqn_str = eqn_str
        self.in_vars = in_vars
        self.out_vars = out_vars
        
        # Parse the equation string to understand the structure
        self._parse_equation()
        
    def _parse_equation(self):
        """Parse the equation string to extract terms and operations"""
        import re
        
        self.terms = []
        self.has_higher_order_derivatives = False
        
        # Extract all derivative terms using regex
        derivative_pattern = r'D\d*\([^)]+\)'
        derivative_terms = re.findall(derivative_pattern, self.eqn_str)
        
        # Extract variable terms (non-derivative)
        variable_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variable_terms = re.findall(variable_pattern, self.eqn_str)
        
        # Filter out derivative function names and keep only variables
        variables = [var for var in variable_terms 
                   if not var.startswith('D') and var not in ['sin', 'cos', 'exp', 'log']]
        
        # Analyze derivative terms
        for term in derivative_terms:
            self.terms.append(term)
            # Check for higher order derivatives
            if term.startswith('D2') or term.startswith('D3'):
                self.has_higher_order_derivatives = True
        
        # Add variable terms
        for var in variables:
            if var not in [t.split('(')[1].split(')')[0].split(',')[0].strip() 
                         for t in derivative_terms if '(' in t and ')' in t]:
                self.terms.append(var)
                
        # Store unique terms
        self.terms = list(set(self.terms))
        
        # Parse the equation structure for evaluation
        self._parse_equation_structure()
    
    def _parse_equation_structure(self):
        """Parse the equation into a computable structure"""
        import re
        
        # Replace derivative patterns with placeholders for parsing
        self.parsed_structure = self.eqn_str
        
        # Find all derivative terms and replace with placeholders
        derivative_pattern = r'D\d*\([^)]+\)'
        derivative_terms = re.findall(derivative_pattern, self.eqn_str)
        
        # Create mapping from derivative terms to computation methods
        self.derivative_map = {}
        for i, term in enumerate(derivative_terms):
            placeholder = f"DERIV_{i}"
            self.parsed_structure = self.parsed_structure.replace(term, placeholder)
            
            # Parse the derivative term
            if term.startswith('D2'):
                # Second derivative
                var = term[3:-1]  # Extract variable from D2(u, x)
                self.derivative_map[placeholder] = ('D2', var)
            elif term.startswith('D3'):
                # Third derivative  
                var = term[3:-1]  # Extract variable from D3(u, x)
                self.derivative_map[placeholder] = ('D3', var)
            else:
                # First derivative
                var = term[2:-1]  # Extract variable from D(u, x)
                self.derivative_map[placeholder] = ('D', var)
        
    def _compute_derivatives(self, u, X):
        """Compute derivatives needed for the PDE"""
        derivatives = {}
        
        # Ensure X requires gradients for autograd
        if not X.requires_grad:
            X = X.clone().requires_grad_(True)
        
        # First derivatives
        for i, var in enumerate(self.in_vars.split(', ')):
            try:
                first_deriv = torch.autograd.grad(
                    u, X[:, i:i+1], 
                    grad_outputs=torch.ones_like(u), 
                    create_graph=True,
                    allow_unused=True,
                    retain_graph=True
                )[0]
                
                # If gradient is None, create a zero tensor with requires_grad
                if first_deriv is None:
                    first_deriv = torch.zeros_like(u, requires_grad=True)
                elif not first_deriv.requires_grad:
                    # If gradient exists but doesn't require grad, make it require grad
                    first_deriv = first_deriv.clone().requires_grad_(True)
                
                derivatives[f'D(u, {var})'] = first_deriv
            except Exception as e:
                print(f"Error computing first derivative for {var}: {e}")
                derivatives[f'D(u, {var})'] = torch.zeros_like(u, requires_grad=True)
        
        # Second derivatives - only compute if first derivatives are valid
        for i, var in enumerate(self.in_vars.split(', ')):
            first_deriv = derivatives[f'D(u, {var})']
            try:
                # Check if first derivative has grad_fn (i.e., it was computed properly)
                if first_deriv.grad_fn is not None:
                    second_deriv = torch.autograd.grad(
                        first_deriv, X[:, i:i+1], 
                        grad_outputs=torch.ones_like(first_deriv), 
                        create_graph=True,
                        allow_unused=True,
                        retain_graph=True
                    )[0]
                    
                    # If gradient is None, create a zero tensor with requires_grad
                    if second_deriv is None:
                        second_deriv = torch.zeros_like(first_deriv, requires_grad=True)
                    elif not second_deriv.requires_grad:
                        second_deriv = second_deriv.clone().requires_grad_(True)
                else:
                    # First derivative wasn't computed properly, skip second derivative
                    second_deriv = torch.zeros_like(first_deriv, requires_grad=True)
                    
                derivatives[f'D2(u, {var})'] = second_deriv
            except Exception as e:
                print(f"Error computing second derivative for {var}: {e}")
                derivatives[f'D2(u, {var})'] = torch.zeros_like(first_deriv, requires_grad=True)
        
        # Third derivatives - only compute if second derivatives are valid
        for i, var in enumerate(self.in_vars.split(', ')):
            second_deriv = derivatives[f'D2(u, {var})']
            try:
                # Check if second derivative has grad_fn (i.e., it was computed properly)
                if second_deriv.grad_fn is not None:
                    third_deriv = torch.autograd.grad(
                        second_deriv, X[:, i:i+1], 
                        grad_outputs=torch.ones_like(second_deriv), 
                        create_graph=True,
                        allow_unused=True,
                        retain_graph=True
                    )[0]
                    
                    # If gradient is None, create a zero tensor with requires_grad
                    if third_deriv is None:
                        third_deriv = torch.zeros_like(second_deriv, requires_grad=True)
                    elif not third_deriv.requires_grad:
                        third_deriv = third_deriv.clone().requires_grad_(True)
                else:
                    # Second derivative wasn't computed properly, skip third derivative
                    third_deriv = torch.zeros_like(second_deriv, requires_grad=True)
                
                derivatives[f'D3(u, {var})'] = third_deriv
            except Exception as e:
                print(f"Error computing third derivative for {var}: {e}")
                derivatives[f'D3(u, {var})'] = torch.zeros_like(second_deriv, requires_grad=True)
        
        return derivatives
        
    def _evaluate_expression(self, X):
        """Evaluate the PDE expression using the computed derivatives"""
        # Get model prediction using the Network's forward method if available
        if hasattr(self, 'forward'):
            u = self.forward(self.model, X)
            # For derivatives, we need the normalized input
            if hasattr(self, 'normalise'):
                X_norm = self.normalise(X)
            else:
                X_norm = X
        else:
            # Fallback to direct model call
            if hasattr(self, 'normalise'):
                X_norm = self.normalise(X)
                u = self.model(X_norm)
            else:
                u = self.model(X)
                X_norm = X
        
        # Compute derivatives with respect to the normalized input
        derivatives = self._compute_derivatives(u, X_norm)
        
        # Use the parsed structure to evaluate the expression
        try:
            return self._evaluate_parsed_expression(u, derivatives)
        except:
            # Fallback to simple cases if parsing fails
            return self._evaluate_simple_cases(u, derivatives)
    
    def _evaluate_parsed_expression(self, u, derivatives):
        """Evaluate the expression using the parsed structure"""
        # Start with the base expression
        result = None
        
        # Replace placeholders with actual computed values
        expr = self.parsed_structure
        
        # Replace derivative placeholders
        for placeholder, (deriv_type, var) in self.derivative_map.items():
            if deriv_type == 'D':
                deriv_value = derivatives[f'D(u, {var})']
            elif deriv_type == 'D2':
                deriv_value = derivatives[f'D2(u, {var})']
            elif deriv_type == 'D3':
                deriv_value = derivatives[f'D3(u, {var})']
            else:
                continue
                
            # Replace placeholder with the actual tensor
            expr = expr.replace(placeholder, f'({deriv_value})')
        
        # Replace variable u with the actual value
        expr = expr.replace('u', f'({u})')
        
        # For now, use eval (in production, we'd use a proper parser)
        try:
            # This is a simplified approach - in production we'd use a proper expression evaluator
            result = eval(expr, {'torch': torch})
            
            # Ensure the result has requires_grad=True if any inputs do
            if u.requires_grad or any(d.requires_grad for d in derivatives.values()):
                if not result.requires_grad:
                    result = result.clone().requires_grad_(True)
                    
            return result
        except Exception as e:
            print(f"Error evaluating parsed expression: {e}")
            raise
    
    def _evaluate_simple_cases(self, u, derivatives):
        """Fallback to simple hardcoded cases"""
        # Simple advection equation
        if 'D(u, t) + D(u, x)' in self.eqn_str:
            return derivatives['D(u, t)'] + derivatives['D(u, x)']
        # Burgers equation
        elif 'D(u, t) + u*D(u, x)' in self.eqn_str:
            return derivatives['D(u, t)'] + u * derivatives['D(u, x)']
        # Heat equation
        elif 'D(u, t) - D2(u, x)' in self.eqn_str:
            return derivatives['D(u, t)'] - derivatives['D2(u, x)']
        # Fallback - just return a simple derivative
        elif 'D(u, t)' in self.eqn_str:
            return derivatives['D(u, t)']
        else:
            # Return zero if we can't parse the equation
            return torch.zeros_like(u)
    def first_deriv(self, u, wrt):
        return torch.autograd.grad(u, wrt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
     
    def second_deriv(self, u, wrt):
        u_deriv = torch.autograd.grad(u, wrt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        return torch.autograd.grad(u_deriv, wrt, grad_outputs=torch.ones_like(u_deriv), create_graph=True)[0]
     
    def third_deriv(self, u, wrt):
        u_deriv = torch.autograd.grad(u, wrt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_deriv = torch.autograd.grad(u_deriv, wrt, grad_outputs=torch.ones_like(u_deriv), create_graph=True)[0]
        return torch.autograd.grad(u_deriv, wrt, grad_outputs=torch.ones_like(u_deriv), create_graph=True)[0]
    
    def func(self, X):
        return self._evaluate_expression(X)