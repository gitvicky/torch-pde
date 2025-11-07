# File-to-File Conversion Mapping

## Package Structure Mapping

| TensorFlow (tf-pde) | PyTorch (torch-pde) | Status | Notes |
|-------------------|-------------------|--------|-------|
| `setup.py` | `setup.py` | ✅ Converted | Updated dependencies |
| `requirements.txt` | `requirements.txt` | ✅ Converted | TF → PyTorch |
| `README.md` | `README.md` | ✅ Updated | PyTorch examples |
| `.gitignore` | `.gitignore` | ✅ Enhanced | Added PyTorch files |

## Core Module Mapping

| TensorFlow Module | PyTorch Module | Key Changes |
|------------------|----------------|-------------|
| `tfpde/__init__.py` | `torchpde/__init__.py` | Package name only |
| `tfpde/main.py` | `torchpde/main.py` | Minimal changes |
| `tfpde/network.py` | `torchpde/network.py` | keras.layers → torch.nn |
| `tfpde/training_ground.py` | `torchpde/training_ground.py` | GradientTape → autograd |
| `tfpde/pde.py` | `torchpde/pde.py` | tf.gradients → torch.autograd.grad |
| `tfpde/boundary_conditions.py` | `torchpde/boundary_conditions.py` | Gradient computation |
| `tfpde/sampler.py` | `torchpde/sampler.py` | Mostly unchanged |
| `tfpde/options.py` | `torchpde/options.py` | Optimizers & activations |
| `tfpde/qnw.py` | `torchpde/qnw.py` | Complete rewrite |
| `tfpde/plotter.py` | `torchpde/plotter.py` | Unchanged |

## Example Files Mapping

| TensorFlow Example | PyTorch Example | Status |
|-------------------|-----------------|--------|
| `Examples/KdV_test.py` | `Examples/KdV_test.py` | ✅ Converted |
| `Examples/Burgers_test.py` | - | ⏳ Not converted yet |
| `Examples/Conv_Diff_test.py` | - | ⏳ Not converted yet |
| `Examples/Advection_test.py` | - | ⏳ Not converted yet |
| `Examples/Diffusion_test.py` | - | ⏳ Not converted yet |

## New Files Added

| File | Purpose |
|------|---------|
| `test_pytorch_pde.py` | Basic functionality test |
| `CONVERSION_NOTES.md` | Detailed technical changes |
| `CONVERSION_SUMMARY.md` | High-level overview |
| `QUICK_MIGRATION_GUIDE.md` | User migration guide |
| `FILE_MAPPING.md` | This document |

## Removed Files

| File | Reason |
|------|--------|
| `tfpde/network_tf_module.py` | Not needed (alternative implementation) |

## Function/Class Correspondence

### Network Module
- `tf.keras.Sequential` → `torch.nn.Sequential`
- `keras.layers.Dense` → `nn.Linear`
- `keras.layers.BatchNormalization` → `nn.BatchNorm1d`
- `tf.keras.initializers.*` → Custom init functions

### Training Module
- `@tf.function` → Regular Python functions
- `tf.GradientTape()` → `loss.backward()`
- `tape.gradient()` → `torch.autograd.grad()`
- `tf.reduce_mean()` → `torch.mean()`
- `tf.square()` → `tensor**2`

### PDE Module
- `tf.gradients()` → `torch.autograd.grad()`
- `tf.concat()` → `torch.cat()`
- `tf.Tensor` → `torch.Tensor`

### Optimizers
- `tf.keras.optimizers.Adam` → `torch.optim.Adam`
- `tfp.optimizer.lbfgs_minimize` → `torch.optim.LBFGS`
- Scipy optimizers → Custom wrapper class

## Data Type Mappings

| TensorFlow | PyTorch |
|------------|---------|
| `tf.float64` | `torch.float64` or `torch.double` |
| `tf.float32` | `torch.float32` or `torch.float` |
| `tf.int32` | `torch.int32` or `torch.int` |
| `dtype=tf.float64` | `dtype=torch.float64` |

## Device Management

| TensorFlow | PyTorch |
|------------|---------|
| Automatic | `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| - | `model.to(device)` |
| - | `tensor.to(device)` |

## Notes
- All core functionality has been preserved
- The API remains virtually identical for end users
- Performance should be comparable or better
- GPU support is more explicit but also more controllable
