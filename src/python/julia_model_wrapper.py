"""
Python wrapper for the Julia BeforeIT model using JuliaCall.

This module provides a direct Python interface to run the Julia BeforeIT model 
using the juliacall package for efficient Python-Julia integration.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import juliacall
    JULIACALL_AVAILABLE = True
except ImportError:
    JULIACALL_AVAILABLE = False
    logger.warning("juliacall not available. Install with: pip install juliacall")


class BeforeITModelWrapper:
    """
    Python wrapper for the Julia BeforeIT model using JuliaCall.
    
    This class provides a direct interface to run the BeforeIT model from Python,
    allowing parameter modifications for calibration purposes.
    """
    
    def __init__(self, 
                 base_parameters: str = "AUSTRIA2010Q1",
                 model_path: str = None,
                 julia_threads: Optional[int] = None):
        """
        Initialize the BeforeIT model wrapper.
        
        Parameters:
        -----------
        base_parameters : str
            Base parameter set to use ("AUSTRIA2010Q1", "ITALY2010Q1", or "STEADY_STATE2010Q1")
        model_path : str
            Path to the BeforeIT.jl package directory
        julia_threads : int, optional
            Number of Julia threads to use (None for auto)
        """
        if not JULIACALL_AVAILABLE:
            raise ImportError("juliacall package not available. Install with: pip install juliacall")
        
        self.base_parameters = base_parameters
        self.model_path = model_path or self._find_model_path()
        
        # Define which parameters can be calibrated
        self.calibratable_parameters = [
            'psi',           # Propensity to consume
            'psi_H',         # Propensity to invest in housing
            'theta_UB',      # Unemployment benefit replacement rate
            'mu',            # Risk premium on policy rate
            'theta_DIV',     # Dividend payout ratio
            'theta',         # Rate of installment on debt
            'zeta',          # Banks' capital requirement coefficient
            'tau_INC',       # Income tax rate
            'tau_FIRM',      # Corporate tax rate
            'tau_VAT',       # Value-added tax rate
        ]
        
        # Initialize Julia environment
        self._setup_julia_environment(julia_threads)
        
    def _find_model_path(self) -> str:
        """Find the path to the BeforeIT.jl package."""
        current_dir = os.getcwd()
        potential_paths = [
            os.path.join(current_dir, "dev", "BeforeIT.jl"),
            os.path.join(current_dir, "..", "..", "dev", "BeforeIT.jl"),
        ]
        
        for path in potential_paths:
            if os.path.exists(os.path.join(path, "src", "BeforeIT.jl")):
                return path
                
        raise FileNotFoundError("Could not find BeforeIT.jl package directory")
    
    def _setup_julia_environment(self, julia_threads: Optional[int]):
        """Set up the Julia environment and import BeforeIT."""
        logger.info("Setting up Julia environment...")
        
        # Get a fresh Julia module to avoid namespace pollution
        self.jl = juliacall.newmodule("BeforeITCalibration")
        
        try:
            # Activate the BeforeIT package environment
            logger.info(f"Activating BeforeIT environment at: {self.model_path}")
            self.jl.seval("using Pkg")
            self.jl.Pkg.activate(self.model_path)
            
            # Install/instantiate all dependencies
            logger.info("Installing package dependencies...")
            self.jl.Pkg.instantiate()
            
            # Import required packages
            logger.info("Importing BeforeIT and required packages...")
            self.jl.seval("import BeforeIT as Bit")
            self.jl.seval("using Random")
            self.jl.seval("using Statistics")
            
            # Test that BeforeIT is working
            self._test_beforeit_import()
            
            logger.info("Julia environment setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Julia environment: {e}")
            raise RuntimeError(f"Julia environment setup failed: {e}")
    
    def _test_beforeit_import(self):
        """Test that BeforeIT can be imported and basic data is available."""
        try:
            # Test that we can access the base parameters
            if self.base_parameters == "AUSTRIA2010Q1":
                params = self.jl.Bit.AUSTRIA2010Q1.parameters
                initial_conditions = self.jl.Bit.AUSTRIA2010Q1.initial_conditions
            elif self.base_parameters == "ITALY2010Q1":
                params = self.jl.Bit.ITALY2010Q1.parameters
                initial_conditions = self.jl.Bit.ITALY2010Q1.initial_conditions
            elif self.base_parameters == "STEADY_STATE2010Q1":
                params = self.jl.Bit.STEADY_STATE2010Q1.parameters
                initial_conditions = self.jl.Bit.STEADY_STATE2010Q1.initial_conditions
            else:
                raise ValueError(f"Unknown base parameters: {self.base_parameters}")
            
            # Check that key parameters exist
            required_params = ['psi', 'tau_INC', 'mu']
            for param in required_params:
                if param not in params:
                    raise KeyError(f"Required parameter '{param}' not found in {self.base_parameters}")
            
            logger.info(f"BeforeIT {self.base_parameters} parameters loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"BeforeIT import test failed: {e}")
    
    def __call__(self, theta: List[float], N: int, rndSeed: int) -> np.ndarray:
        """
        Run the BeforeIT model with given parameters.
        
        This is the main interface required by black-it calibration.
        
        Parameters:
        -----------
        theta : List[float]
            Parameter values to calibrate
        N : int
            Number of time periods to simulate
        rndSeed : int
            Random seed for simulation
            
        Returns:
        --------
        np.ndarray
            Simulated time series data of shape (N, 9)
        """
        return self.run_simulation(theta, N, rndSeed)
    
    def run_simulation(self, 
                      parameters: List[float], 
                      T: int, 
                      seed: int,
                      ensemble_size: int = 4) -> np.ndarray:
        """
        Run a BeforeIT simulation with specified parameters.
        
        Parameters:
        -----------
        parameters : List[float]
            Parameter values corresponding to self.calibratable_parameters
        T : int
            Number of time periods to simulate
        seed : int
            Random seed
        ensemble_size : int
            Number of ensemble runs to average
            
        Returns:
        --------
        np.ndarray
            Simulated time series data of shape (T, 9)
        """
        if len(parameters) != len(self.calibratable_parameters):
            raise ValueError(f"Expected {len(self.calibratable_parameters)} parameters, "
                           f"got {len(parameters)}")
        
        logger.info(f"Running simulation with T={T}, ensemble_size={ensemble_size}, seed={seed}")
        logger.debug(f"Parameters: {dict(zip(self.calibratable_parameters, parameters))}")
        
        try:
            # Set random seed in Julia
            getattr(self.jl.Random, "seed!")(seed)
            
            # Get base parameters and initial conditions
            if self.base_parameters == "AUSTRIA2010Q1":
                base_params = self.jl.copy(self.jl.Bit.AUSTRIA2010Q1.parameters)
                initial_conditions = self.jl.Bit.AUSTRIA2010Q1.initial_conditions
            elif self.base_parameters == "ITALY2010Q1":
                base_params = self.jl.copy(self.jl.Bit.ITALY2010Q1.parameters)
                initial_conditions = self.jl.Bit.ITALY2010Q1.initial_conditions
            elif self.base_parameters == "STEADY_STATE2010Q1":
                base_params = self.jl.copy(self.jl.STEADY_STATE2010Q1.parameters)
                initial_conditions = self.jl.Bit.STEADY_STATE2010Q1.initial_conditions
            else:
                raise ValueError(f"Unknown base parameters: {self.base_parameters}")
            
            # Update parameters with calibration values
            for param_name, param_value in zip(self.calibratable_parameters, parameters):
                if param_name in base_params:
                    base_params[param_name] = param_value
                else:
                    logger.warning(f"Parameter {param_name} not found in base parameters")
            
            # Run ensemble simulation
            model = self.jl.Bit.Model(base_params, initial_conditions)
            model_vec = self.jl.Bit.ensemblerun(model, T, ensemble_size)
            
            # Get available variables dynamically
            variable_names = self._get_available_variables(model)
            
            result_array = np.zeros((T, len(variable_names)))
            
            for i, var_name in enumerate(variable_names):
                # Collect data from all ensemble runs
                var_data_list = []
                for model_instance in model_vec:
                    var_values = getattr(model_instance.data, var_name)
                    # Convert Julia array to Python/numpy and take last T periods
                    var_array = np.array(var_values)
                    var_data_list.append(var_array[-T:])
                
                # Stack and average across ensemble runs
                if var_data_list:
                    var_matrix = np.column_stack(var_data_list)
                    result_array[:, i] = np.mean(var_matrix, axis=1)
                else:
                    logger.warning(f"No data found for variable {var_name}")
            
            logger.info("Simulation completed successfully")
            return result_array
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise RuntimeError(f"Julia simulation failed: {e}")
    
    def run_single_simulation(self, 
                             parameters: List[float], 
                             T: int, 
                             seed: int) -> np.ndarray:
        """
        Run a single BeforeIT simulation (no ensemble).
        
        Useful for testing or when ensemble averaging is not needed.
        
        Parameters:
        -----------
        parameters : List[float]
            Parameter values corresponding to self.calibratable_parameters
        T : int
            Number of time periods to simulate
        seed : int
            Random seed
            
        Returns:
        --------
        np.ndarray
            Simulated time series data of shape (T, 9)
        """
        if len(parameters) != len(self.calibratable_parameters):
            raise ValueError(f"Expected {len(self.calibratable_parameters)} parameters, "
                           f"got {len(parameters)}")
        
        logger.info(f"Running single simulation with T={T}, seed={seed}")
        
        try:
            # Set random seed in Julia
            getattr(self.jl.Random, "seed!")(seed)
            
            # Get base parameters and initial conditions
            if self.base_parameters == "AUSTRIA2010Q1":
                base_params = self.jl.copy(self.jl.Bit.AUSTRIA2010Q1.parameters)
                initial_conditions = self.jl.Bit.AUSTRIA2010Q1.initial_conditions
            elif self.base_parameters == "ITALY2010Q1":
                base_params = self.jl.copy(self.jl.Bit.ITALY2010Q1.parameters)
                initial_conditions = self.jl.Bit.ITALY2010Q1.initial_conditions
            elif self.base_parameters == "STEADY_STATE2010Q1":
                base_params = self.jl.copy(self.jl.Bit.STEADY_STATE2010Q1.parameters)
                initial_conditions = self.jl.Bit.STEADY_STATE2010Q1.initial_conditions
            else:
                raise ValueError(f"Unknown base parameters: {self.base_parameters}")
            
            # Update parameters with calibration values
            for param_name, param_value in zip(self.calibratable_parameters, parameters):
                if param_name in base_params:
                    base_params[param_name] = param_value
            
            # Create and run model
            model = self.jl.Bit.Model(base_params, initial_conditions)
            
            # Run simulation step by step
            for t in range(T):
                getattr(self.jl.Bit, "step!")(model, multi_threading=False)
                getattr(self.jl.Bit, "update_data!")(model)
            
            # Get available variables dynamically
            variable_names = self._get_available_variables(model)
            
            result_array = np.zeros((T, len(variable_names)))
            
            for i, var_name in enumerate(variable_names):
                var_values = getattr(model.data, var_name)
                var_array = np.array(var_values)
                # Take the last T periods to match the requested simulation length
                result_array[:, i] = var_array[-T:]
            
            logger.info("Single simulation completed successfully")
            return result_array
            
        except Exception as e:
            logger.error(f"Single simulation failed: {e}")
            raise RuntimeError(f"Julia simulation failed: {e}")
    
    def get_parameter_bounds(self) -> List[List[float]]:
        """
        Get reasonable bounds for calibratable parameters.
        
        Returns:
        --------
        List[List[float]]
            List of [min_value, max_value] pairs for each parameter
        """
        bounds = {
            'psi': [0.4, 0.9],           # Propensity to consume
            'psi_H': [0.01, 0.15],       # Propensity to invest in housing  
            'theta_UB': [0.3, 0.8],      # Unemployment benefit replacement rate
            'mu': [0.005, 0.05],         # Risk premium on policy rate
            'theta_DIV': [0.2, 0.8],     # Dividend payout ratio
            'theta': [0.05, 0.25],       # Rate of installment on debt
            'zeta': [0.05, 0.15],        # Banks' capital requirement coefficient
            'tau_INC': [0.1, 0.4],       # Income tax rate
            'tau_FIRM': [0.15, 0.35],    # Corporate tax rate
            'tau_VAT': [0.15, 0.25],     # Value-added tax rate
        }
        
        return [bounds[param] for param in self.calibratable_parameters]
    
    def get_parameter_precisions(self) -> List[float]:
        """
        Get precision levels for calibratable parameters.
        
        Returns:
        --------
        List[float]
            Precision values for each parameter
        """
        precisions = {
            'psi': 0.01,           
            'psi_H': 0.001,        
            'theta_UB': 0.01,      
            'mu': 0.001,           
            'theta_DIV': 0.01,     
            'theta': 0.01,         
            'zeta': 0.001,         
            'tau_INC': 0.01,       
            'tau_FIRM': 0.01,      
            'tau_VAT': 0.01,       
        }
        
        return [precisions[param] for param in self.calibratable_parameters]
    
    def get_parameter_names(self) -> List[str]:
        """Get the names of calibratable parameters."""
        return self.calibratable_parameters.copy()
    
    def get_current_parameter_values(self) -> Dict[str, float]:
        """
        Get current parameter values from the base parameter set.
        
        Returns:
        --------
        Dict[str, float]
            Current parameter values
        """
        try:
            if self.base_parameters == "AUSTRIA2010Q1":
                params = self.jl.Bit.AUSTRIA2010Q1.parameters
            elif self.base_parameters == "ITALY2010Q1":
                params = self.jl.Bit.ITALY2010Q1.parameters
            elif self.base_parameters == "STEADY_STATE2010Q1":
                params = self.jl.Bit.STEADY_STATE2010Q1.parameters
            else:
                raise ValueError(f"Unknown base parameters: {self.base_parameters}")
            
            current_values = {}
            for param_name in self.calibratable_parameters:
                if param_name in params:
                    current_values[param_name] = float(params[param_name])
                else:
                    logger.warning(f"Parameter {param_name} not found in base parameters")
                    current_values[param_name] = 0.0
            
            return current_values
            
        except Exception as e:
            logger.error(f"Failed to get current parameter values: {e}")
            return {param: 0.0 for param in self.calibratable_parameters}

    def get_variable_names(self) -> List[str]:
        """Get the names of variables which are contained in the model.data object."""
        try:
            # Access Data type field names directly - no model instance needed!
            fieldnames = self.jl.fieldnames(self.jl.Bit.Data)
            
            # Convert Julia symbols to Python strings
            return [str(field) for field in fieldnames]
            
        except Exception as e:
            logger.error(f"Failed to get variable names: {e}")
            # Return a fallback list of common variables
            return ["real_gdp", "real_household_consumption", "real_government_consumption",
                   "real_capitalformation", "real_exports", "real_imports", 
                   "wages", "euribor", "gdp_deflator_growth_ea"]

    def _get_available_variables(self, model=None) -> List[str]:
        """Helper method to get available variables from the Data type."""
        try:
            # Access Data type field names directly - no model instance needed!
            fieldnames = self.jl.fieldnames(self.jl.Bit.Data)
            # Convert Julia symbols to Python strings and filter out non-data fields
            all_fields = [str(field) for field in fieldnames]
            
            # Define variables we're interested in for economic analysis
            # These are based on the actual Data struct from BeforeIT
            desired_variables = [
                "real_gdp", "real_household_consumption", "real_government_consumption",
                "real_capitalformation", "real_exports", "real_imports", 
                "wages", "euribor", "gdp_deflator_growth_ea", "real_gdp_ea",
                "nominal_gdp", "nominal_household_consumption", "nominal_government_consumption",
                "nominal_capitalformation", "nominal_exports", "nominal_imports",
                "operating_surplus", "compensation_employees", "taxes_production"
            ]
            
            # Return intersection of available and desired variables
            available_variables = [var for var in desired_variables if var in all_fields]
            
            if not available_variables:
                # If no desired variables found, return all available fields except collection_time
                logger.warning("No desired economic variables found, returning all available fields")
                available_variables = [field for field in all_fields if field != "collection_time"]
            
            logger.info(f"Using {len(available_variables)} variables: {available_variables}")
            return available_variables
            
        except Exception as e:
            logger.error(f"Failed to get available variables: {e}")
            # Return fallback
            return ["real_gdp", "real_household_consumption", "real_government_consumption"]

def test_model_wrapper():
    """Test the model wrapper with default parameters."""
    print("Testing BeforeIT model wrapper with JuliaCall...")
    
    try:
        # Initialize wrapper
        print("Initializing wrapper...")
        wrapper = BeforeITModelWrapper()
        
        # Test parameter methods
        bounds = wrapper.get_parameter_bounds()
        names = wrapper.get_parameter_names()
        precisions = wrapper.get_parameter_precisions()
        current_values = wrapper.get_current_parameter_values()
        
        print(f"✓ Found {len(names)} calibratable parameters")
        print(f"✓ Parameter bounds and precisions loaded")
                
        # Show current parameter values
        print("\nCurrent parameter values:")
        for name, value in current_values.items():
            print(f"  {name:12s}: {value:.6f}")
        
        # Test with default parameters (approximately middle of bounds)
        test_params = [(b[0] + b[1]) / 2 for b in bounds]
        print(f"\nTesting with middle-of-bounds parameters...")
        
        # Run short simulation
        print("Running test simulation...")
        result = wrapper.run_single_simulation(test_params, T=3, seed=42)
        
        print(f"✓ Simulation completed with output shape: {result.shape}")
        
        # Get the actual variable names from the wrapper
        variable_names = wrapper.get_variable_names()
        print(f"Simulation returns the following variables: {variable_names}")
        
        print("First few values for each variable:")
        for i, var_name in enumerate(variable_names[:result.shape[1]]):  # Only show variables that exist in result
            print(f"  {var_name:25s}: {result[:, i]}")
        
        # Test ensemble simulation
        print("\nTesting ensemble simulation...")
        ensemble_result = wrapper.run_simulation(test_params, T=3, seed=42, ensemble_size=2)
        print(f"✓ Ensemble simulation completed with output shape: {ensemble_result.shape}")
        
        print("\n✅ Model wrapper test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_model_wrapper() 