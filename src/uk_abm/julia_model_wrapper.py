"""
Python wrapper for the Julia BeforeIT model using JuliaCall.

This module provides a direct Python interface to run the Julia BeforeIT model
using the juliacall package for efficient Python-Julia integration.
"""

import numpy as np
from typing import List, Dict, Optional
import logging
import os

from .model_results import VariableType, ModelVariable, ModelResults

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

    def __init__(
        self,
        base_parameters: str = "AUSTRIA2010Q1",
        model_path: str = None,
        julia_threads: Optional[int] = None,
        excluded_parameters: Optional[List[str]] = None,
    ):
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
        excluded_parameters : List[str], optional
            List of parameter names to exclude from calibration (useful for structural parameters)
        """
        if not JULIACALL_AVAILABLE:
            raise ImportError(
                "juliacall package not available. Install with: pip install juliacall"
            )

        self.base_parameters = base_parameters
        self.model_path = model_path or self._find_model_path()
        
        # Set default excluded parameters - these are typically structural or data-driven
        # and shouldn't be calibrated
        default_excluded = excluded_parameters or [
            "C",  # Input-output matrix (structural)
            "C_G",  # Government consumption matrix (data-driven)
            "C_E",  # Firm consumption matrix (data-driven)
            "c_E_g",  # Government expenditure coefficients (data-driven)
            "w_s",  # Sector wages (data-driven)
            "N_s",  # Sector employment (data-driven)
            "Y",  # Sector output (data-driven)
            "Y_I",  # Investment output (data-driven)
            "Y_EA",  # EA output (data-driven)
            "Y_EA_series",  # EA output series (data-driven)
            "pi",  # Sector-specific inflation (data-driven)
            "H_act",  # Active households (structural)
            "alpha_pi_EA",  # EA inflation coefficient (estimated elsewhere)
            "xi_pi",  # Inflation persistence (estimated elsewhere)
        ]
        self.excluded_parameters = set(default_excluded)

        # Initialize Julia environment
        self._setup_julia_environment(julia_threads)
        
        # Get calibratable parameters dynamically from the model
        self.calibratable_parameters = self._get_calibratable_parameters()

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
            required_params = ["psi", "tau_INC", "mu"]
            for param in required_params:
                if param not in params:
                    raise KeyError(
                        f"Required parameter '{param}' not found in {self.base_parameters}"
                    )

            logger.info(
                f"BeforeIT {self.base_parameters} parameters loaded successfully"
            )

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
            Simulated time series data of shape (N, n_variables)
        """
        results = self.run_simulation(theta, N, rndSeed)
        return results.to_calibration_array()

    def run_simulation(
        self, parameters: List[float], T: int, seed: int, ensemble_size: int = 4
    ) -> ModelResults:
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
        ModelResults
            Structured results with flexible extraction methods
        """
        if len(parameters) != len(self.calibratable_parameters):
            raise ValueError(
                f"Expected {len(self.calibratable_parameters)} parameters, "
                f"got {len(parameters)}"
            )

        logger.info(
            f"Running simulation with T={T}, ensemble_size={ensemble_size}, seed={seed}"
        )
        logger.debug(
            f"Parameters: {dict(zip(self.calibratable_parameters, parameters))}"
        )

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
            for param_name, param_value in zip(
                self.calibratable_parameters, parameters
            ):
                if param_name in base_params:
                    base_params[param_name] = param_value
                else:
                    logger.warning(
                        f"Parameter {param_name} not found in base parameters"
                    )

            # Run ensemble simulation
            model = self.jl.Bit.Model(base_params, initial_conditions)
            model_vec = self.jl.Bit.ensemblerun(model, T, ensemble_size)

            # Create structured results
            results = ModelResults.from_julia_model(
                model_vec, T, self, ensemble_size=ensemble_size
            )
            
            # Add simulation metadata
            results.simulation_metadata = {
                "parameters": dict(zip(self.calibratable_parameters, parameters)),
                "seed": seed,
                "T": T,
                "ensemble_size": ensemble_size,
                "base_parameters": self.base_parameters
            }

            logger.info("Simulation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise RuntimeError(f"Julia simulation failed: {e}")

    def run_single_simulation(
        self, parameters: List[float], T: int, seed: int
    ) -> ModelResults:
        """
        Run a single BeforeIT simulation (no ensemble).

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
        ModelResults
            Structured results with flexible extraction methods
        """
        if len(parameters) != len(self.calibratable_parameters):
            raise ValueError(
                f"Expected {len(self.calibratable_parameters)} parameters, "
                f"got {len(parameters)}"
            )

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
            for param_name, param_value in zip(
                self.calibratable_parameters, parameters
            ):
                if param_name in base_params:
                    base_params[param_name] = param_value

            # Create and run model
            model = self.jl.Bit.Model(base_params, initial_conditions)

            # Run simulation step by step
            for t in range(T):
                getattr(self.jl.Bit, "step!")(model, multi_threading=False)
                getattr(self.jl.Bit, "update_data!")(model)

            # Create structured results
            results = ModelResults.from_julia_model(
                model, T, self, ensemble_size=1
            )
            
            # Add simulation metadata
            results.simulation_metadata = {
                "parameters": dict(zip(self.calibratable_parameters, parameters)),
                "seed": seed,
                "T": T,
                "ensemble_size": 1,
                "base_parameters": self.base_parameters
            }

            logger.info("Single simulation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Single simulation failed: {e}")
            raise RuntimeError(f"Julia simulation failed: {e}")

    def get_parameter_bounds(self) -> np.ndarray:
        """
        Get reasonable bounds for calibratable parameters.

        Returns:
        --------
        np.ndarray
            2D array of shape [2, n_params] where bounds[0,:] are lower bounds 
            and bounds[1,:] are upper bounds
        """
        # Predefined bounds for common parameters
        known_bounds = {
            "psi": [0.4, 0.9],  # Propensity to consume
            "psi_H": [0.01, 0.15],  # Propensity to invest in housing
            "theta_UB": [0.3, 0.8],  # Unemployment benefit replacement rate
            "mu": [0.005, 0.05],  # Risk premium on policy rate
            "theta_DIV": [0.2, 0.8],  # Dividend payout ratio
            "theta": [0.05, 0.25],  # Rate of installment on debt
            "zeta": [0.05, 0.15],  # Banks' capital requirement coefficient
            "tau_INC": [0.1, 0.4],  # Income tax rate
            "tau_FIRM": [0.15, 0.35],  # Corporate tax rate
            "tau_VAT": [0.15, 0.25],  # Value-added tax rate
            "tau_SIF": [0.1, 0.4],  # Social insurance contribution rate (firms)
            "tau_SIW": [0.05, 0.2],  # Social insurance contribution rate (workers)
            "rho": [0.8, 0.99],  # Persistence parameter
            "xi": [0.01, 0.1],  # Adjustment parameter
            "beta": [0.8, 0.99],  # Discount factor
            "gamma": [0.5, 2.0],  # Risk aversion/elasticity parameter
        }

        bounds = []
        current_values = self.get_current_parameter_values()
        
        for param in self.calibratable_parameters:
            if param in known_bounds:
                bounds.append(known_bounds[param])
            else:
                # For unknown parameters, create bounds around current value
                current_val = current_values.get(param, 0.1)
                if current_val > 0:
                    # For positive parameters, use ±50% of current value
                    lower = max(0.001, current_val * 0.5)
                    upper = current_val * 1.5
                else:
                    # For parameters that could be negative, use symmetric bounds
                    abs_val = abs(current_val) if current_val != 0 else 0.1
                    lower = -abs_val * 1.5
                    upper = abs_val * 1.5
                bounds.append([lower, upper])
                logger.info(f"Generated bounds for {param}: [{lower:.4f}, {upper:.4f}]")

        # Convert to numpy array with shape [2, n_params]
        bounds_array = np.array(bounds).T  # Transpose to get [2, n_params]
        return bounds_array

    def get_parameter_precisions(self) -> np.ndarray:
        """
        Get precision levels for calibratable parameters.

        Returns:
        --------
        np.ndarray
            1D array of precision values for each parameter
        """
        # Predefined precisions for common parameters
        known_precisions = {
            "psi": 0.01,
            "psi_H": 0.001,
            "theta_UB": 0.01,
            "mu": 0.001,
            "theta_DIV": 0.01,
            "theta": 0.01,
            "zeta": 0.001,
            "tau_INC": 0.01,
            "tau_FIRM": 0.01,
            "tau_VAT": 0.01,
            "tau_SIF": 0.01,
            "tau_SIW": 0.01,
            "rho": 0.001,
            "xi": 0.001,
            "beta": 0.001,
            "gamma": 0.01,
        }

        precisions = []
        bounds = self.get_parameter_bounds()
        
        for i, param in enumerate(self.calibratable_parameters):
            if param in known_precisions:
                precisions.append(known_precisions[param])
            else:
                # For unknown parameters, use 1% of the parameter range as precision
                param_range = bounds[1, i] - bounds[0, i]
                precision = param_range * 0.01
                precisions.append(precision)
                logger.info(f"Generated precision for {param}: {precision:.6f}")

        return np.array(precisions)

    def get_parameter_names(self) -> List[str]:
        """Get the names of calibratable parameters."""
        return self.calibratable_parameters.copy()

    def add_excluded_parameters(self, param_names: List[str]):
        """
        Add parameters to the excluded list and refresh calibratable parameters.
        
        Parameters:
        -----------
        param_names : List[str]
            Parameter names to exclude from calibration
        """
        self.excluded_parameters.update(param_names)
        self.calibratable_parameters = self._get_calibratable_parameters()
        logger.info(f"Added {param_names} to excluded parameters. "
                   f"Now have {len(self.calibratable_parameters)} calibratable parameters.")

    def remove_excluded_parameters(self, param_names: List[str]):
        """
        Remove parameters from the excluded list and refresh calibratable parameters.
        
        Parameters:
        -----------
        param_names : List[str]
            Parameter names to allow for calibration
        """
        self.excluded_parameters.difference_update(param_names)
        self.calibratable_parameters = self._get_calibratable_parameters()
        logger.info(f"Removed {param_names} from excluded parameters. "
                   f"Now have {len(self.calibratable_parameters)} calibratable parameters.")

    def set_excluded_parameters(self, param_names: List[str]):
        """
        Set the excluded parameters list and refresh calibratable parameters.
        
        Parameters:
        -----------
        param_names : List[str]
            Complete list of parameter names to exclude from calibration
        """
        self.excluded_parameters = set(param_names)
        self.calibratable_parameters = self._get_calibratable_parameters()
        logger.info(f"Set excluded parameters to {param_names}. "
                   f"Now have {len(self.calibratable_parameters)} calibratable parameters.")

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
                    logger.warning(
                        f"Parameter {param_name} not found in base parameters"
                    )
                    current_values[param_name] = 0.0

            return current_values

        except Exception as e:
            logger.error(f"Failed to get current parameter values: {e}")
            return {param: 0.0 for param in self.calibratable_parameters}

    def get_variable_names(self, include_aggregated_panels: bool = False) -> List[str]:
        """Get the names of variables which are contained in the model.data object."""
        try:
            fieldnames = self.jl.fieldnames(self.jl.Bit.Data)
            
            # Filter our the collection time
            fieldnames = [field for field in fieldnames if str(field) != "collection_time"]
            
            # Return
            if include_aggregated_panels:
                return [str(field) for field in fieldnames]
            else:
                return [
                    str(field)
                    for field in fieldnames
                    if "sector" not in str(field)
                ]

        except Exception as e:
            logger.error(f"Failed to get variable names: {e}")
            # Return a fallback list of common variables
            return [
                "real_gdp",
                "real_household_consumption",
                "real_government_consumption",
                "real_capitalformation",
                "real_exports",
                "real_imports",
                "wages",
                "euribor",
                "gdp_deflator_growth_ea",
            ]

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
                "real_gdp",
                "real_household_consumption",
                "real_government_consumption",
                "real_capitalformation",
                "real_exports",
                "real_imports",
                "wages",
                "euribor",
                "gdp_deflator_growth_ea",
                "real_gdp_ea",
                "nominal_gdp",
                "nominal_household_consumption",
                "nominal_government_consumption",
                "nominal_capitalformation",
                "nominal_exports",
                "nominal_imports",
                "operating_surplus",
                "compensation_employees",
                "taxes_production",
            ]

            # Return intersection of available and desired variables
            available_variables = [
                var for var in desired_variables if var in all_fields
            ]

            if not available_variables:
                # If no desired variables found, return all available fields except collection_time
                logger.warning(
                    "No desired economic variables found, returning all available fields"
                )
                available_variables = [
                    field for field in all_fields if field != "collection_time"
                ]

            logger.info(
                f"Using {len(available_variables)} variables: {available_variables}"
            )
            return available_variables

        except Exception as e:
            logger.error(f"Failed to get available variables: {e}")
            # Return fallback
            return [
                "real_gdp",
                "real_household_consumption",
                "real_government_consumption",
            ]

    def _get_calibratable_parameters(self) -> List[str]:
        """
        Dynamically get the names of parameters that can be calibrated.
        
        Returns all parameter keys from the base parameter set, excluding
        those in the excluded_parameters list.
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

            # Get all parameter names from the Julia dict
            all_param_names = list(params.keys())
            
            # Filter out excluded parameters and data structures (arrays, matrices)
            calibratable = []
            for param_name in all_param_names:
                if param_name not in self.excluded_parameters:
                    # Check if parameter is a scalar (not an array/matrix)
                    param_value = params[param_name]
                    try:
                        # Convert to numpy to check if it's a scalar
                        param_array = np.array(param_value)
                        if param_array.ndim == 0:  # Scalar parameter
                            calibratable.append(param_name)
                        else:
                            logger.debug(f"Excluding non-scalar parameter: {param_name}")
                    except Exception:
                        # If conversion fails, assume it's not a simple numeric parameter
                        logger.debug(f"Excluding non-numeric parameter: {param_name}")

            logger.info(f"Found {len(calibratable)} calibratable parameters: {calibratable}")
            return calibratable

        except Exception as e:
            logger.error(f"Failed to get calibratable parameters: {e}")
            # Fallback to a basic set if dynamic detection fails
            return ["psi", "theta_UB", "mu", "tau_INC", "tau_FIRM", "tau_VAT"]


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
        print("✓ Parameter bounds and precisions loaded")

        # Show current parameter values
        print("\nCurrent parameter values:")
        for name, value in current_values.items():
            print(f"  {name:12s}: {value:.6f}")

        # Test with default parameters (approximately middle of bounds)
        test_params = (bounds[0, :] + bounds[1, :]) / 2
        print("\nTesting with middle-of-bounds parameters...")

        # Run short simulation
        print("Running test simulation...")
        result = wrapper.run_single_simulation(test_params, T=3, seed=42)

        print(f"✓ Simulation completed - ModelResults with {len(result.variables)} variables")
        print(f"  • Time series variables: {len(result.list_timeseries_variables())}")
        print(f"  • Panel data variables: {len(result.list_panel_variables())}")

        # Convert to calibration array for demonstration
        calibration_array = result.to_calibration_array()
        print(f"  • Calibration array shape: {calibration_array.shape}")

        print("Sample time series data:")
        timeseries_data = result.get_scalar_timeseries()
        for i, (var_name, data) in enumerate(timeseries_data.items()):
            if i < 3:  # Show first 3 time series
                print(f"  {var_name:25s}: {data}")

        if result.list_panel_variables():
            print(f"Panel data available: {result.list_panel_variables()[:2]}...")

        # Test ensemble simulation
        print("\nTesting ensemble simulation...")
        ensemble_result = wrapper.run_simulation(
            test_params, T=3, seed=42, ensemble_size=2
        )
        ensemble_array = ensemble_result.to_calibration_array()
        print(f"✓ Ensemble simulation completed")
        print(f"  • Ensemble ModelResults with {len(ensemble_result.variables)} variables")
        print(f"  • Ensemble calibration array shape: {ensemble_array.shape}")
        print(f"  • Simulation metadata: {ensemble_result.simulation_metadata}")

        print("\n✅ Model wrapper test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Model wrapper test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_model_wrapper()
