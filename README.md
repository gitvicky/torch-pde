# Neural PDE Solver Python Package : torch-pde
Automatic Differentiation based Partial Differential Equation solver implemented on PyTorch. Package distribution under the MIT License. Built for students to get initiated on Neural PDE Solvers as described in the paper [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

### Installation 

Since the package was built as a proof-of-concept, support for it has been discontinued. However the package still works with the mentioned dependencies. We suggest running the package within a conda environment. 

```python
conda create -n TorchPDE python=3.8
conda activate TorchPDE
pip install torch-pde
```

### Key Changes from TensorFlow Version

- **Framework**: Converted from TensorFlow 2.x to PyTorch
- **Automatic Differentiation**: Uses `torch.autograd` instead of `tf.gradients`
- **Optimizers**: Supports PyTorch optimizers and scipy optimizers
- **GPU Support**: Automatic GPU detection and usage when available
- **Precision**: Uses double precision (float64) by default

### [Example(s)](https://github.com/gitvicky/torch-pde/tree/master/Examples)
To solve a particular PDE using a PINN, the package requires information on the three parameters: neural network hyperparameters, sampling parameters, information about the PDE and the case that we are solving for : 

```python
import torchpde 

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

```
---
Partial derivative of y with respect to x is represented by D(y, x) and the second order derivative is given by D(D(y, x), x) or D2(y, x).
 
---
These parameters are used to initialise the model and sample the training data: 


```python
model = torchpde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)
```

Once the model is initiated, we determine the training parameters and solve for the PDE: 


```python
train_config = {'Optimizer': 'adam',
                 'learning_rate': 0.001, 
                 'Iterations' : 50000}

training_time = model.train(train_config, training_data)
```

The PDE solution can be extracted by running a feedforward operation of the trained network and compared with traditional numerical methods: 


```python
u_pred = model.predict(X_star)
```

### GPU Support

The PyTorch version automatically detects and uses GPU if available:

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
```

### Optimizer Options

The PyTorch version supports various optimizers:

- **Gradient Descent**: 'adam', 'sgd', 'adamw', 'rmsprop', 'adagrad', 'adadelta', 'adamax'
- **Quasi-Newton**: 'LBFGS', 'L-BFGS'  
- **Scipy Optimizers**: 'L-BFGS-B', 'BFGS', 'Nelder-Mead', 'Powell', 'CG', 'TNC', 'COBYLA', 'SLSQP'
