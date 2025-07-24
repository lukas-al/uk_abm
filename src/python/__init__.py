"""BeforeIT Model Calibration Package with JuliaCall"""

__version__ = "2.0.0"
__author__ = "BeforeIT Calibration Team" 
__license__ = "AGPL-3.0"

try:
    import juliacall
    from .calibrate_beforeit import BeforeITCalibrator
    from .data_loader import get_calibration_targets, plot_target_data
    from .julia_model_wrapper import BeforeITModelWrapper
    
    __all__ = ['BeforeITCalibrator', 'BeforeITModelWrapper', 
               'get_calibration_targets', 'plot_target_data']

except ImportError as e:
    raise ImportError(f"Missing dependencies: {e}. Install with: pip install juliacall black-it")