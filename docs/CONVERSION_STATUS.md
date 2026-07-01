# Torch-PDE Conversion Project

## Goals
1. Convert the existing TensorFlow-based PDE solver to PyTorch
2. Maintain the same interface and functionality
3. Update to use latest PyTorch versions and best practices
4. Ensure all examples work with the new implementation

## Current Status

### Completed
- Created new PyTorch-based directory structure
- Rewrote core modules with PyTorch equivalents:
  - `network.py`: Replaced TensorFlow Keras with PyTorch nn.Module
  - `pde.py`: Updated symbolic differentiation to use PyTorch autograd
  - `boundary_conditions.py`: Converted to PyTorch tensors and autograd
  - `sampler.py`: Maintained sampling logic, updated for PyTorch compatibility
  - `options.py`: Updated optimizers and initializers for PyTorch
  - `main.py`: Maintained interface, updated internal calls
  - `training_ground.py`: Core training logic converted to PyTorch
  - `qnw.py`: Quasi-Newton wrapper updated for PyTorch
  - `plotter.py`: Maintained plotting functionality

### In Progress
- Testing basic functionality
- Debugging import issues (pyDOE missing)

### Next Steps
1. Fix pyDOE import issue
2. Test basic model creation and prediction
3. Run simple PDE examples to verify correctness
4. Compare results with original TensorFlow implementation
5. Optimize performance and memory usage
6. Document any interface changes or improvements

## Key Changes Made

### Architecture Changes
- Replaced TensorFlow Keras layers with PyTorch nn.Module
- Updated autograd from TensorFlow to PyTorch
- Modified optimizer handling for PyTorch
- Updated tensor operations to PyTorch equivalents

### API Compatibility
- Maintained all original function signatures
- Preserved parameter names and structures
- Kept the same training and prediction interfaces

## Testing Plan

1. **Unit Testing**: Test individual components (network, PDE, boundary conditions)
2. **Integration Testing**: Test full training pipeline
3. **Example Testing**: Run all original examples and compare results
4. **Performance Testing**: Compare training speed and memory usage

## Known Issues
- pyDOE import error needs to be resolved
- Need to verify gradient computation correctness
- Training loop needs thorough testing
- Quasi-Newton optimizers need validation

## Future Improvements
- Add proper error handling
- Implement logging
- Add type hints
- Create comprehensive documentation
- Add more examples and tutorials