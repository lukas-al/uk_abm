"""
Data structures for handling BeforeIT model simulation results.

This module provides flexible data containers for simulation output that can handle
both 1D time series and multidimensional panel data from the Julia model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VariableType(Enum):
    """Enum for different types of model variables."""
    SCALAR_TIMESERIES = "scalar_timeseries"  # 1D time series
    PANEL = "panel"                          # 2D arrays (sectors x time, etc.)
    HIGHER_DIM = "higher_dim"                # 3D+ arrays 
    SCALAR = "scalar"                        # Single values
    UNKNOWN = "unknown"                      # Couldn't determine type


@dataclass
class ModelVariable:
    """Container for a single model variable with metadata and conversion utilities."""
    name: str
    data: np.ndarray
    variable_type: VariableType
    original_shape: tuple
    time_periods: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the variable after creation."""
        if self.data.size == 0:
            logger.warning(f"Variable {self.name} has no data")
    
    def is_timeseries(self) -> bool:
        """Check if this is a 1D time series."""
        return self.variable_type == VariableType.SCALAR_TIMESERIES
    
    def is_panel(self) -> bool:
        """Check if this is panel/cross-sectional data."""
        return self.variable_type == VariableType.PANEL
    
    def aggregate(self, method: str = "sum", axis: Optional[int] = None) -> np.ndarray:
        """
        Aggregate multidimensional data to 1D time series.
        
        Parameters:
        -----------
        method : str
            Aggregation method: 'sum', 'mean', 'max', 'min'
        axis : int, optional
            Axis to aggregate over. If None, auto-detect time axis.
            
        Returns:
        --------
        np.ndarray
            1D time series of length time_periods
        """
        if self.variable_type == VariableType.SCALAR_TIMESERIES:
            return self.data
        elif self.variable_type == VariableType.SCALAR:
            # Broadcast scalar to time series
            return np.full(self.time_periods, self.data.item())
        elif self.variable_type == VariableType.PANEL:
            if axis is None:
                # Auto-detect time axis vs. cross-sectional axis
                if self.data.shape[0] == self.time_periods:
                    axis = 1  # Sum across columns (sectors/units)
                    logger.debug(f"Auto-detected time axis=0 for {self.name}, aggregating across axis=1")
                elif self.data.shape[1] == self.time_periods:
                    axis = 0  # Sum across rows (sectors/units)  
                    logger.debug(f"Auto-detected time axis=1 for {self.name}, aggregating across axis=0")
                else:
                    logger.warning(f"Neither dimension of {self.name} matches time_periods={self.time_periods}, shape={self.data.shape}")
                    axis = -1  # Aggregate over last dimension
            
            if method == "sum":
                result = np.sum(self.data, axis=axis)
            elif method == "mean":
                result = np.mean(self.data, axis=axis)
            elif method == "max":
                result = np.max(self.data, axis=axis)
            elif method == "min":
                result = np.min(self.data, axis=axis)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
                
            # Ensure we return the right length
            if len(result) >= self.time_periods:
                return result[-self.time_periods:]
            else:
                return np.pad(result, (self.time_periods - len(result), 0), 'constant')
        else:
            # For higher-dim or unknown, flatten and extract time series
            logger.warning(f"Aggregating {self.variable_type} variable {self.name} by flattening")
            flat = self.data.flatten()
            if len(flat) >= self.time_periods:
                return flat[-self.time_periods:]
            else:
                return np.pad(flat, (self.time_periods - len(flat), 0), 'constant')


@dataclass  
class ModelResults:
    """Container for all results from a BeforeIT model simulation with flexible extraction."""
    variables: Dict[str, ModelVariable] = field(default_factory=dict)
    time_periods: int = 0
    ensemble_size: int = 1
    simulation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_julia_model(cls, julia_model_or_ensemble, time_periods: int, 
                        wrapper_instance, ensemble_size: int = 1) -> 'ModelResults':
        """Create ModelResults from Julia model output."""
        results = cls(time_periods=time_periods, ensemble_size=ensemble_size)
        
        # Get variable names from Julia Data struct
        try:
            fieldnames = wrapper_instance.jl.fieldnames(wrapper_instance.jl.Bit.Data) 
            variable_names = [str(field) for field in fieldnames]
        except Exception as e:
            logger.error(f"Failed to get variable names: {e}")
            variable_names = []
        
        # Process each variable
        for var_name in variable_names:
            if var_name == "collection_time":  # Skip metadata fields
                continue
                
            try:
                # Detect if we have a single model or a vector of models
                # Check if julia_model_or_ensemble has a .data attribute (single model)
                # or if it's iterable (vector of models)
                try:
                    # Try to access .data directly (single model case)
                    single_model_data = julia_model_or_ensemble.data
                    # If we get here, it's a single model
                    julia_data = getattr(single_model_data, var_name)
                    results.add_variable_from_julia(var_name, julia_data, time_periods)
                except (AttributeError, TypeError):
                    # No .data attribute, so it must be a vector of models
                    var_data_list = []
                    for model_instance in julia_model_or_ensemble:
                        julia_data = getattr(model_instance.data, var_name)
                        var_array = results._convert_julia_to_numpy(julia_data, var_name)
                        var_data_list.append(var_array)
                    
                    # Average across ensemble runs if multiple, or just use single value
                    if len(var_data_list) == 1:
                        results._add_numpy_variable(var_name, var_data_list[0], time_periods)
                    elif var_data_list:
                        # Multiple runs - average them
                        shapes = [arr.shape for arr in var_data_list]
                        if len(set(shapes)) == 1:
                            # All same shape - can stack and average
                            stacked = np.stack(var_data_list, axis=0)
                            averaged = np.mean(stacked, axis=0)
                            results._add_numpy_variable(var_name, averaged, time_periods)
                        else:
                            logger.warning(f"Inconsistent shapes for {var_name} across ensemble: {shapes}")
                            # Take first one as fallback
                            results._add_numpy_variable(var_name, var_data_list[0], time_periods)
                        
            except Exception as e:
                logger.warning(f"Failed to process variable {var_name}: {e}")
                
        return results
    
    def add_variable_from_julia(self, name: str, julia_data, time_periods: int, metadata: Dict = None):
        """Convert Julia data and add as ModelVariable."""
        numpy_data = self._convert_julia_to_numpy(julia_data, name)
        self._add_numpy_variable(name, numpy_data, time_periods, metadata)
    
    def _convert_julia_to_numpy(self, julia_data, var_name: str) -> np.ndarray:
        """Convert Julia data to numpy array with robust error handling."""
        try:
            # Try direct conversion first
            data_list = list(julia_data)
            return np.array(data_list, dtype=float)
        except Exception as e:
            logger.debug(f"Direct conversion failed for {var_name}: {e}")
            try:
                # Try element-wise conversion for nested structures
                if hasattr(julia_data, '__len__') and len(julia_data) > 0:
                    # Check if it's a nested structure (like Vector{Vector{Float64}})
                    first_element = julia_data[0] if len(julia_data) > 0 else None
                    if hasattr(first_element, '__len__'):
                        # It's nested - convert to 2D array
                        nested_list = []
                        for i in range(len(julia_data)):
                            inner_list = list(julia_data[i])
                            nested_list.append(inner_list)
                        return np.array(nested_list, dtype=float)
                    else:
                        # Regular 1D conversion
                        data_list = [float(julia_data[i]) for i in range(len(julia_data))]
                        return np.array(data_list)
                else:
                    # Single value
                    return np.array([float(julia_data)])
            except Exception as e2:
                logger.error(f"All conversion attempts failed for {var_name}: {e2}")
                # Final fallback
                return np.array([0.0])
    
    def _add_numpy_variable(self, name: str, data: np.ndarray, time_periods: int, metadata: Dict = None):
        """Add a numpy array as a ModelVariable with automatic type detection."""
        # Determine variable type based on shape
        if data.ndim == 0:
            var_type = VariableType.SCALAR
        elif data.ndim == 1:
            var_type = VariableType.SCALAR_TIMESERIES
        elif data.ndim == 2:
            var_type = VariableType.PANEL
        else:
            var_type = VariableType.HIGHER_DIM
        
        variable = ModelVariable(
            name=name,
            data=data,
            variable_type=var_type,
            original_shape=data.shape,
            time_periods=time_periods,
            metadata=metadata or {}
        )
        
        self.variables[name] = variable
        logger.debug(f"Added variable {name}: {var_type.value}, shape {data.shape}")
    
    def get_scalar_timeseries(self) -> Dict[str, np.ndarray]:
        """Get all 1D time series variables."""
        return {
            name: var.data 
            for name, var in self.variables.items() 
            if var.variable_type == VariableType.SCALAR_TIMESERIES
        }
    
    def get_panel_data(self) -> Dict[str, np.ndarray]:
        """Get all panel/2D variables with their raw structure."""
        return {
            name: var.data 
            for name, var in self.variables.items() 
            if var.variable_type == VariableType.PANEL
        }
    
    def get_aggregated_panels(self, method: str = "sum", 
                            variables: List[str] = None,
                            axis: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get panel data aggregated to 1D time series.
        
        Parameters:
        -----------
        method : str
            Aggregation method ('sum', 'mean', 'max', 'min')
        variables : List[str], optional
            Specific panel variables to aggregate. If None, aggregate all.
        axis : int, optional
            Axis to aggregate over. If None, auto-detect.
        """
        panel_vars = self.get_panel_data()
        if variables:
            panel_vars = {k: v for k, v in panel_vars.items() if k in variables}
        
        return {
            name: self.variables[name].aggregate(method=method, axis=axis)
            for name in panel_vars.keys()
        }
    
    def to_calibration_array(self, 
                           variables: List[str] = None, 
                           include_aggregated_panels: bool = False,
                           panel_aggregation: str = "sum",
                           panel_axis: Optional[int] = None) -> np.ndarray:
        """
        Convert to format needed for calibration (T x N_variables array).
        
        Parameters:
        -----------
        variables : List[str], optional
            Specific variables to include. If None, include all suitable variables.
        include_aggregated_panels : bool
            Whether to include panel data aggregated to time series
        panel_aggregation : str
            How to aggregate panel data ('sum', 'mean', etc.)
        panel_axis : int, optional
            Axis to aggregate panel data over
            
        Returns:
        --------
        np.ndarray
            Array of shape (time_periods, n_variables)
        """
        
        # Get 1D time series
        scalar_vars = self.get_scalar_timeseries()
        
        # Optionally include aggregated panel data
        if include_aggregated_panels:
            aggregated_panels = self.get_aggregated_panels(
                method=panel_aggregation, 
                variables=variables,
                axis=panel_axis
            )
            scalar_vars.update(aggregated_panels)
        
        # Filter to requested variables if specified
        if variables:
            scalar_vars = {k: v for k, v in scalar_vars.items() if k in variables}
        
        if not scalar_vars:
            logger.warning("No variables available for calibration array")
            return np.array([])
        
        # Ensure all have same length (time_periods)
        processed_vars = {}
        for name, data in scalar_vars.items():
            if len(data) >= self.time_periods:
                processed_vars[name] = data[-self.time_periods:]
            else:
                # Pad if too short
                padded = np.pad(data, (self.time_periods - len(data), 0), 'constant')
                processed_vars[name] = padded
                logger.warning(f"Padded variable {name} from length {len(data)} to {self.time_periods}")
        
        # Stack into array with consistent ordering
        var_names = sorted(processed_vars.keys())  
        arrays = [processed_vars[name] for name in var_names]
        
        if arrays:
            result = np.column_stack(arrays)
            logger.info(f"Created calibration array with shape {result.shape} for variables: {var_names}")
            return result
        else:
            return np.array([])
    
    def get_variable_summary(self) -> Dict[str, Dict]:
        """Get summary information about all variables for inspection."""
        summary = {}
        for name, var in self.variables.items():
            summary[name] = {
                "type": var.variable_type.value,
                "shape": var.original_shape,
                "has_data": var.data.size > 0,
                "data_range": (float(np.min(var.data)), float(np.max(var.data))) if var.data.size > 0 else (None, None),
                "metadata": var.metadata
            }
        return summary
    
    def list_panel_variables(self) -> List[str]:
        """Get names of all panel/cross-sectional variables."""
        return [name for name, var in self.variables.items() if var.is_panel()]
    
    def list_timeseries_variables(self) -> List[str]:
        """Get names of all 1D time series variables."""
        return [name for name, var in self.variables.items() if var.is_timeseries()] 