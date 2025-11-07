# Quick Migration Guide: tf-pde to torch-pde

## Installation

### Old (TensorFlow)
```bash
pip install tensorflow==2.2.0 tensorflow-probability==0.10.0
```

### New (PyTorch)
```bash
pip install torch>=1.9.0
```

## Code Changes

### 1. Import Statement
```python
# Old
import tfpde

# New
import torchpde
```

### 2. Everything Else Stays the Same!
```python
# These remain IDENTICAL in both versions:

# Neural Network setup
NN_parameters = {
    'Network_Type': 'Regular',
    'input_neurons': 2,
    'output_neurons': 1,
    'num_layers': 4,
    'num_neurons': 64
}

# PDE setup  
PDE_parameters = {
    'Inputs': 't, x',
    'Outputs': 'u',
    'Equation': 'D(u, t) + u*D(u, x) + 0.0025*D3(u, x)',
    'lower_range': [0.0, -1.0],
    'upper_range': [1.0, 1.0],
    'Boundary_Condition': "Periodic",
    'Boundary_Vals': None,
    'Initial_Condition': lambda x: np.cos(np.pi*x),
    'Initial_Vals': None
}

# Model creation
model = torchpde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)

# Training
train_config = {'Optimizer': 'adam', 'learning_rate': 0.001, 'Iterations': 5000}
model.train(train_config, training_data)

# Prediction
u_pred = model.predict(X_star)
```

## GPU Usage

### TensorFlow (automatic)
```python
# GPU used automatically if available
```

### PyTorch (automatic with info)
```python
# GPU used automatically if available
# You can check:
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

## That's It!
The API is identical - just change the import statement and you're good to go!
