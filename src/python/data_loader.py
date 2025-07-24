"""
Data loader for BeforeIT model calibration.

This module generates synthetic or loads real target data for calibrating the BeforeIT model.
For now, it creates synthetic data that matches the expected structure and scale of the model outputs.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt


def generate_synthetic_target_data(T: int = 20, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic target data for BeforeIT model calibration.
    
    The BeforeIT model outputs 9 main time series:
    - real_gdp
    - real_household_consumption  
    - real_government_consumption
    - real_capitalformation
    - real_exports
    - real_imports
    - wages
    - euribor
    - gdp_deflator
    
    Parameters:
    -----------
    T : int
        Number of time periods
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Array of shape (T, 9) containing synthetic time series data
    """
    np.random.seed(seed)
    
    # Base values roughly matching Austrian economy scale (in millions EUR for GDP components)
    base_values = {
        'real_gdp': 134000,  # Base real GDP
        'real_household_consumption': 75000,  # ~55% of GDP
        'real_government_consumption': 25000,  # ~19% of GDP  
        'real_capitalformation': 28000,  # ~21% of GDP
        'real_exports': 67000,  # ~50% of GDP (Austria is export-oriented)
        'real_imports': 61000,  # ~45% of GDP
        'wages': 58000,  # Compensation of employees
        'euribor': 0.015,  # 1.5% interest rate
        'gdp_deflator': 1.02,  # 2% inflation trend
    }
    
    # Trend growth rates (quarterly)
    growth_rates = {
        'real_gdp': 0.003,  # ~1.2% annual growth
        'real_household_consumption': 0.002,
        'real_government_consumption': 0.001,
        'real_capitalformation': 0.004,  # More volatile
        'real_exports': 0.005,  # Export growth
        'real_imports': 0.004,
        'wages': 0.003,
        'euribor': 0.0,  # Mean reverting around base level
        'gdp_deflator': 0.005,  # Quarterly inflation ~2% annual
    }
    
    # Volatility (standard deviation of innovations)
    volatilities = {
        'real_gdp': 0.008,
        'real_household_consumption': 0.006,
        'real_government_consumption': 0.004,
        'real_capitalformation': 0.015,  # More volatile
        'real_exports': 0.012,
        'real_imports': 0.010,
        'wages': 0.005,
        'euribor': 0.002,  # Interest rate volatility
        'gdp_deflator': 0.003,
    }
    
    variables = list(base_values.keys())
    data = np.zeros((T, len(variables)))
    
    # Generate time series with trends and business cycle components
    for i, var in enumerate(variables):
        base = base_values[var]
        trend = growth_rates[var]
        vol = volatilities[var]
        
        # Initialize first period
        data[0, i] = base
        
        # Generate subsequent periods with AR(1) structure and trend
        ar_coeff = 0.7  # Persistence in business cycle
        
        for t in range(1, T):
            # Trend component
            trend_component = base * (1 + trend) ** t
            
            # Business cycle component (AR(1) around trend)
            if t == 1:
                cycle_component = np.random.normal(0, vol * base)
            else:
                cycle_component = ar_coeff * (data[t-1, i] - base * (1 + trend) ** (t-1)) + \
                                np.random.normal(0, vol * base)
            
            data[t, i] = trend_component + cycle_component
            
            # Ensure positive values for real variables
            if var not in ['euribor']:
                data[t, i] = max(data[t, i], base * 0.5)  # Minimum bound
    
    return data


def load_real_uk_data(start_date: str = "2010Q1", periods: int = 20) -> np.ndarray:
    """
    Load real UK economic data for calibration.
    
    TODO: Implement loading real UK data from ONS or other sources.
    For now, returns synthetic data.
    
    Parameters:
    -----------
    start_date : str
        Start date in YYYY-QX format
    periods : int
        Number of quarters to load
        
    Returns:
    --------
    np.ndarray
        Array of shape (periods, 9) containing real time series data
    """
    print("Warning: Real UK data loading not implemented yet. Using synthetic data.")
    return generate_synthetic_target_data(periods)


def get_variable_names() -> list:
    """
    Get the names of the 9 model output variables.
    
    Returns:
    --------
    list
        Variable names in order
    """
    return [
        'real_gdp',
        'real_household_consumption',
        'real_government_consumption', 
        'real_capitalformation',
        'real_exports',
        'real_imports',
        'wages',
        'euribor',
        'gdp_deflator'
    ]


def plot_target_data(data: np.ndarray, save_path: str = None):
    """
    Plot the target data for visualization.
    
    Parameters:
    -----------
    data : np.ndarray
        Target data array of shape (T, 9)
    save_path : str, optional
        Path to save the plot
    """
    variables = get_variable_names()
    T = data.shape[0]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Target Data for BeforeIT Model Calibration', fontsize=16)
    
    for i, (var, ax) in enumerate(zip(variables, axes.flat)):
        ax.plot(range(T), data[:, i], 'b-', linewidth=2)
        ax.set_title(var.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Time Period')
        ax.grid(True, alpha=0.3)
        
        # Special formatting for interest rates
        if var == 'euribor':
            ax.set_ylabel('Rate')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3%}'))
        elif var == 'gdp_deflator':
            ax.set_ylabel('Index')
        else:
            ax.set_ylabel('Millions EUR')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def get_calibration_targets(data_source: str = "synthetic", **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Get calibration target data and metadata.
    
    Parameters:
    -----------
    data_source : str
        Data source: "synthetic" or "uk_real"
    **kwargs
        Additional arguments passed to data loading functions
        
    Returns:
    --------
    tuple
        (target_data, metadata) where target_data is shape (T, 9) and 
        metadata contains information about the data
    """
    if data_source == "synthetic":
        T = kwargs.get('T', 20)
        seed = kwargs.get('seed', 42)
        data = generate_synthetic_target_data(T, seed)
        metadata = {
            'source': 'synthetic',
            'periods': T,
            'seed': seed,
            'variables': get_variable_names(),
            'description': 'Synthetic data for BeforeIT model calibration'
        }
    elif data_source == "uk_real":
        start_date = kwargs.get('start_date', "2010Q1")
        periods = kwargs.get('periods', 20)
        data = load_real_uk_data(start_date, periods)
        metadata = {
            'source': 'uk_real',
            'start_date': start_date,
            'periods': periods,
            'variables': get_variable_names(),
            'description': 'Real UK economic data'
        }
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    
    return data, metadata


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic target data...")
    target_data, metadata = get_calibration_targets("synthetic", T=40, seed=42)
    
    print(f"Target data shape: {target_data.shape}")
    print(f"Variables: {metadata['variables']}")
    
    # Plot the data
    plot_target_data(target_data, "plots/target_data_plot.png")
    
    # Save data for calibration
    np.save("data/target_data.npy", target_data)
    print("Target data saved to target_data.npy") 