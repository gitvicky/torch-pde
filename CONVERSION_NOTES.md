# TensorFlow to PyTorch Conversion - Key Differences

## Overview
This document outlines the key changes made when converting the tf-pde package from TensorFlow 2.x to PyTorch.

## Major Framework Changes

### 1. Automatic Differentiation
- **TensorFlow**: `tf.gradients(y, x)`
- **PyTorch**: `torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))`

### 2. Neural Network Definition
- **TensorFlow**: Uses `tf.keras.Sequential` and `tf.keras.layers`
- **PyTorch**: Uses `torch.nn.Sequential` and `torch.nn` modules

### 3. Tensor Operations
- **TensorFlow**: Uses `tf.Tensor` with eager execution
- **PyTorch**: Uses `torch.Tensor` with automatic differentiation

### 4. Device Management
- **TensorFlow**: Implicit GPU usage
- **PyTorch**: Explicit device management with `.to(device)`

## Code Structure Changes

### Network Module
```python
# TensorFlow version
model = keras.Sequential()
model.add(keras.layers.Dense(units, activation=activation))

# PyTorch version
layers_list = []
layers_list.append(nn.Linear(in_features, out_features))
layers_list.append(activation)
model = nn.Sequential(*layers_list)
```

### Gradient Computation
```python
# TensorFlow version
@tf.function
def loss_and_gradients(self, X_i, u_i, X_b, u_b, X_f):
    with tf.GradientTape() as tape:
        model_loss = self.loss_func(X_i, u_i, X_b, u_b, X_f)
    model_gradients = tape.gradient(model_loss, self.trainable_params)
    
# PyTorch version
def train_step(self, X_i, u_i, X_b, u_b, X_f):
    optimizer.zero_grad()
    loss = self.loss_func(X_i, u_i, X_b, u_b, X_f)
    loss.backward()
    optimizer.step()
```

### PDE Derivatives
```python
# TensorFlow version
def first_deriv(self, u, wrt):
    return tf.gradients(u, wrt)[0]

# PyTorch version  
def first_deriv(self, u, wrt):
    return autograd.grad(u, wrt, 
                        grad_outputs=torch.ones_like(u),
                        retain_graph=True,
                        create_graph=True)[0]
```

## Optimizer Changes

### Standard Optimizers
- Both frameworks support Adam, SGD, RMSprop, etc.
- PyTorch adds AdamW (Adam with weight decay)
- Syntax differs but functionality is similar

### L-BFGS Implementation
- **TensorFlow**: Uses `tfp.optimizer.lbfgs_minimize` from TensorFlow Probability
- **PyTorch**: Uses native `torch.optim.LBFGS` with closure function

### Scipy Integration
- Both versions support scipy optimizers
- PyTorch version uses a wrapper class to interface between PyTorch tensors and numpy arrays

## Performance Considerations

### GPU Support
- **TensorFlow**: Automatic GPU usage when available
- **PyTorch**: Explicit GPU management provides more control
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

### Memory Management
- PyTorch requires explicit gradient management with `requires_grad=True`
- PyTorch provides `torch.no_grad()` context for inference

### Precision
- Both versions use float64 (double precision) by default
- PyTorch: `torch.set_default_dtype(torch.float64)`
- TensorFlow: `tf.keras.backend.set_floatx('float64')`

## Testing and Validation

To ensure the conversion is correct:

1. **Numerical Accuracy**: Both versions should produce similar results for the same PDE
2. **Performance**: Training time should be comparable
3. **GPU Utilization**: Both should effectively use GPU when available
4. **Optimizer Convergence**: Similar convergence behavior for the same optimizer

## Usage Differences

The API remains largely the same:

```python
# Both versions use the same setup
model = torchpde.main.setup(NN_parameters, NPDE_parameters, PDE_parameters)

# Training is identical
train_config = {'Optimizer': 'adam', 'learning_rate': 0.001, 'Iterations': 5000}
model.train(train_config, training_data)

# Prediction is the same
u_pred = model.predict(X_star)
```

## Advantages of PyTorch Version

1. **Better GPU Control**: Explicit device management
2. **Dynamic Graphs**: More flexible for research
3. **Native LBFGS**: Built-in L-BFGS optimizer
4. **Pythonic**: More intuitive for Python developers
5. **Active Development**: Rapidly evolving ecosystem

## Migration Guide

To migrate from tf-pde to torch-pde:

1. Install PyTorch instead of TensorFlow
2. Replace `import tfpde` with `import torchpde`  
3. No changes needed to problem setup dictionaries
4. Training and prediction code remains the same
5. For custom losses, use PyTorch autograd instead of tf.GradientTape
