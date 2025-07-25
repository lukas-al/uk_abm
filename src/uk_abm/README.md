# BeforeIT Model Calibration with Black-IT

This package provides a Python interface for calibrating the BeforeIT economic agent-based model using the [black-it](https://bancaditalia.github.io/black-it/) calibration toolbox and [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) for efficient Python-Julia integration.

## Overview

The BeforeIT model is a sophisticated agent-based economic model developed by the Bank of Italy. This calibration system allows you to:

- Automatically calibrate key model parameters using real or synthetic data
- Use advanced optimization algorithms for parameter search
- Visualize calibration progress and parameter space exploration
- Compare model outputs with target data

## Features

- **Multiple Sampling Algorithms**: Uses a combination of uniform sampling, Halton sequences, random forests, and Gaussian processes
- **Flexible Data Sources**: Support for synthetic target data and real economic data
- **Comprehensive Visualization**: Automatic generation of calibration progress plots, parameter exploration charts, and model comparison graphs
- **Efficient Julia Integration**: Direct Python-Julia communication via JuliaCall (no subprocess overhead)
- **Extensible Design**: Easy to add new parameters, loss functions, or data sources

## Installation

### Prerequisites

1. **Julia**: Install Julia (version 1.6 or higher) from [julialang.org](https://julialang.org/downloads/)
2. **BeforeIT.jl**: The Julia BeforeIT package should be installed (included in this repository under `dev/BeforeIT.jl`)

### Python Dependencies

Install the required Python packages:

```bash
pip install juliacall black-it numpy matplotlib pandas scipy scikit-learn optuna
```

Or install from the requirements file:

```bash
pip install -r ../../requirements.txt
```

## Quick Start

### 1. Basic Calibration

```python
from calibrate_beforeit import BeforeITCalibrator

# Initialize calibrator with synthetic target data
calibrator = BeforeITCalibrator(
    target_data_source="synthetic",
    base_parameters="AUSTRIA2010Q1"
)

# Run calibration
best_params, best_losses, all_params, all_losses = calibrator.run_calibration(
    max_iterations=10,
    ensemble_size=4,
    batch_size=6
)

# Analyze and visualize results
calibrator.analyze_results()
```

### 2. Command Line Usage

```bash
python calibrate_beforeit.py
```

This will run a calibration with default settings and save results to a timestamped folder.

### 3. Testing the Setup

```bash
python test_julia_integration.py
```

This runs comprehensive tests to verify that JuliaCall and BeforeIT are properly configured.

## Components

### 1. Data Loader (`data_loader.py`)

Handles target data generation and loading:

- **Synthetic Data**: Generates realistic economic time series with trends and business cycles
- **Real Data**: Framework for loading real economic data (UK data loader to be implemented)
- **Visualization**: Plotting functions for target data

```python
from data_loader import get_calibration_targets, plot_target_data

# Load synthetic target data
target_data, metadata = get_calibration_targets("synthetic", T=20, seed=42)
plot_target_data(target_data)
```

### 2. Julia Model Wrapper (`julia_model_wrapper.py`)

Provides direct Python interface to the Julia BeforeIT model using JuliaCall:

- **Direct Integration**: No subprocess overhead, direct function calls
- **Parameter Management**: Handles parameter bounds, precisions, and validation
- **Simulation Control**: Manages ensemble runs and random seeding
- **Type Conversion**: Automatic conversion between Python and Julia types
- **Persistent Session**: Reuses Julia environment for faster repeated calls

```python
from julia_model_wrapper import BeforeITModelWrapper

# Initialize wrapper (sets up Julia environment)
wrapper = BeforeITModelWrapper(base_parameters="AUSTRIA2010Q1")

# Get parameter information
bounds = wrapper.get_parameter_bounds()
names = wrapper.get_parameter_names()
current_values = wrapper.get_current_parameter_values()

# Run simulation directly
params = [0.6, 0.05, 0.5, 0.02, 0.4, 0.1, 0.08, 0.25, 0.25, 0.2]  # Example values
result = wrapper.run_simulation(params, T=20, seed=42)

# Or run single simulation (no ensemble)
result = wrapper.run_single_simulation(params, T=20, seed=42)
```

### 3. Main Calibrator (`calibrate_beforeit.py`)

Orchestrates the calibration process:

- **Sampler Configuration**: Sets up multiple optimization algorithms
- **Loss Function**: Uses Method of Simulated Moments by default
- **Progress Tracking**: Monitors and visualizes calibration progress
- **Results Analysis**: Comprehensive post-calibration analysis

## Architecture

### Julia Integration via JuliaCall

The system uses [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) for seamless Python-Julia integration:

```python
import juliacall

# Create isolated Julia module
jl = juliacall.newmodule("BeforeITCalibration")

# Activate BeforeIT environment
jl.seval("using Pkg")
jl.Pkg.activate("/path/to/BeforeIT.jl")

# Import BeforeIT directly
jl.seval("import BeforeIT as Bit")

# Direct function calls
model = jl.Bit.Model(parameters, initial_conditions)
results = jl.Bit.ensemblerun(model, T, ensemble_size)
```

## Calibrated Parameters

The system calibrates the following 10 key parameters:

| Parameter | Description | Bounds |
|-----------|-------------|--------|
| `psi` | Propensity to consume | [0.4, 0.9] |
| `psi_H` | Propensity to invest in housing | [0.01, 0.15] |
| `theta_UB` | Unemployment benefit replacement rate | [0.3, 0.8] |
| `mu` | Risk premium on policy rate | [0.005, 0.05] |
| `theta_DIV` | Dividend payout ratio | [0.2, 0.8] |
| `theta` | Rate of installment on debt | [0.05, 0.25] |
| `zeta` | Banks' capital requirement coefficient | [0.05, 0.15] |
| `tau_INC` | Income tax rate | [0.1, 0.4] |
| `tau_FIRM` | Corporate tax rate | [0.15, 0.35] |
| `tau_VAT` | Value-added tax rate | [0.15, 0.25] |

## Model Outputs

The system calibrates against 9 macroeconomic time series:

1. **Real GDP** (millions EUR)
2. **Real Household Consumption** (millions EUR)
3. **Real Government Consumption** (millions EUR)
4. **Real Capital Formation** (millions EUR)
5. **Real Exports** (millions EUR)
6. **Real Imports** (millions EUR)
7. **Wages** (millions EUR)
8. **Euribor** (interest rate)
9. **GDP Deflator** (price index)

## Configuration

### Basic Configuration

```python
config = {
    'target_data_source': 'synthetic',  # or 'uk_real'
    'base_parameters': 'AUSTRIA2010Q1',  # or 'ITALY2010Q1', 'STEADY_STATE2010Q1'
    'max_iterations': 15,
    'ensemble_size': 4,
    'batch_size': 6,
    'T': 20,  # Number of time periods
    'seed': 42,
}
```

### Advanced Configuration

```python
# Custom parameter bounds
wrapper = BeforeITModelWrapper()
custom_bounds = wrapper.get_parameter_bounds()
custom_bounds[0] = [0.5, 0.8]  # Tighter bounds for psi

# Custom loss function
from black_it.loss_functions.minkowski import MinkowskiLoss
loss_func = MinkowskiLoss(p=2)  # Euclidean distance

# Different samplers
from black_it.samplers.random_uniform import RandomUniformSampler
samplers = [RandomUniformSampler(batch_size=8)]

# Julia threading (optional)
wrapper = BeforeITModelWrapper(julia_threads=4)
```

## Output and Results

The calibration process generates several outputs:

### Files Created
- `target_data.png` - Visualization of target data
- `calibration_progress.png` - Loss convergence and parameter evolution
- `parameter_exploration.png` - Parameter space exploration
- `target_vs_simulation.png` - Comparison of best fit vs target
- `best_parameters.npy` - Best parameter values
- `calibration_results.npy` - Complete results data
- `results_summary.txt` - Human-readable summary

### Folder Structure
```
calibration_results_YYYYMMDD_HHMMSS/
├── target_data.png
├── calibration_progress.png
├── parameter_exploration.png
├── target_vs_simulation.png
├── best_parameters.npy
├── best_simulation.npy
├── calibration_results.npy
└── results_summary.txt
```

## Performance Optimization

### Julia Threading

Set the number of Julia threads for better performance:

```python
# Option 1: Environment variable (before importing juliacall)
import os
os.environ['PYTHON_JULIACALL_THREADS'] = '4'

# Option 2: Via wrapper initialization
wrapper = BeforeITModelWrapper(julia_threads=4)
```

### Calibration Performance Tips

1. **Ensemble Size**: Start with smaller ensembles (2-4) for initial exploration
2. **Batch Size**: Larger batches are more efficient for parameter space exploration
3. **Early Stopping**: Monitor loss convergence to avoid unnecessary iterations
4. **Memory Management**: The Julia session persists, so large models stay in memory

## Troubleshooting

### Common Issues

1. **JuliaCall Installation**
   ```bash
   pip install juliacall
   ```

2. **BeforeIT.jl Package Not Found**
   - Ensure the `dev/BeforeIT.jl` directory exists
   - Check that `src/BeforeIT.jl` is present in the package

3. **Julia Environment Issues**
   - JuliaCall automatically manages Julia dependencies
   - It will download Julia if not available
   - Check Julia package installation with `test_julia_integration.py`

4. **Memory Issues**
   - Reduce `ensemble_size` or `batch_size`
   - Use shorter time series (`T`)
   - Restart Python to clear Julia session

5. **Performance Issues**
   - Increase Julia threads via environment variable
   - Use `run_single_simulation()` for testing
   - Monitor memory usage during long calibrations

### Testing the Setup

Test the JuliaCall integration:
```python
from julia_model_wrapper import test_model_wrapper
test_model_wrapper()
```

Test all components:
```python
python test_julia_integration.py
```

Run example calibrations:
```python
python example_calibration.py
```

## Extending the System

### Adding New Parameters

1. Add parameter name to `calibratable_parameters` in `BeforeITModelWrapper`
2. Add bounds and precision in the respective methods
3. The parameter will automatically be passed to Julia

### Adding New Data Sources

1. Create a new loading function in `data_loader.py`
2. Add the data source option to `get_calibration_targets()`
3. Ensure the output format matches (T, 9) array

### Custom Loss Functions

```python
from black_it.loss_functions.base import BaseLoss
import numpy as np

class CustomLoss(BaseLoss):
    def compute_loss(self, sim_data, real_data):
        # Your custom loss calculation
        return np.mean((sim_data - real_data) ** 2)

# Use in calibration
calibrator.run_calibration(loss_function=CustomLoss())
```

### Custom Julia Functions

You can extend the Julia integration by adding custom functions:

```python
# Access the Julia module directly
jl = wrapper.jl

# Define custom Julia function
jl.seval("""
function my_custom_analysis(model)
    # Custom Julia code here
    return some_result
end
""")

# Call the custom function
result = jl.my_custom_analysis(model)
```

## Citation

If you use this calibration system in your research, please cite:

- The BeforeIT model: [Economic forecasting with an agent-based model](https://www.sciencedirect.com/science/article/pii/S0014292122001891)
- The black-it toolbox: [Black-it documentation](https://bancaditalia.github.io/black-it/)
- JuliaCall: [PythonCall & JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/)

## License

This calibration system is provided under the same license as the BeforeIT.jl package (AGPL-3.0).

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Run `python test_julia_integration.py` for diagnostic information
3. Review the BeforeIT.jl documentation
4. Consult the black-it documentation
5. Check JuliaCall documentation for integration issues
6. Open an issue in the repository 