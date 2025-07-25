"""
Data loader for BeforeIT model calibration.

This module generates synthetic or loads real target data for calibrating the BeforeIT model.
The synthetic data generation is now model-aware and uses variable names from the model.
"""

import numpy as np
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt


def get_variable_metadata() -> Dict[str, Dict]:
    """
    Get metadata for generating synthetic economic variables.
    
    Returns mapping: variable_name -> {base_value, growth_rate, volatility, type, unit}
    """
    return {
        "real_gdp": {
            "base_value": 134000, 
            "growth_rate": 0.003, 
            "volatility": 0.008,
            "type": "gdp_component",
            "unit": "millions_eur"
        },
        "real_household_consumption": {
            "base_value": 75000,
            "growth_rate": 0.002, 
            "volatility": 0.006,
            "type": "gdp_component",
            "unit": "millions_eur"
        },
        "real_government_consumption": {
            "base_value": 25000,
            "growth_rate": 0.001, 
            "volatility": 0.004,
            "type": "gdp_component",
            "unit": "millions_eur"
        },
        "real_capitalformation": {
            "base_value": 28000,
            "growth_rate": 0.004, 
            "volatility": 0.015,
            "type": "gdp_component",
            "unit": "millions_eur"
        },
        "real_exports": {
            "base_value": 67000,
            "growth_rate": 0.005, 
            "volatility": 0.012,
            "type": "gdp_component",
            "unit": "millions_eur"
        },
        "real_imports": {
            "base_value": 61000,
            "growth_rate": 0.004, 
            "volatility": 0.010,
            "type": "gdp_component",
            "unit": "millions_eur"
        },
        "wages": {
            "base_value": 58000,
            "growth_rate": 0.003, 
            "volatility": 0.005,
            "type": "income",
            "unit": "millions_eur"
        },
        "euribor": {
            "base_value": 0.015,
            "growth_rate": 0.0, 
            "volatility": 0.002,
            "type": "interest_rate",
            "unit": "rate"
        },
        "gdp_deflator": {
            "base_value": 1.02,
            "growth_rate": 0.005, 
            "volatility": 0.003,
            "type": "price_index",
            "unit": "index"
        },
        "gdp_deflator_growth_ea": {
            "base_value": 0.02,
            "growth_rate": 0.0, 
            "volatility": 0.003,
            "type": "growth_rate",
            "unit": "rate"
        },
        "nominal_gdp": {
            "base_value": 136500,
            "growth_rate": 0.008, 
            "volatility": 0.010,
            "type": "nominal_gdp_component",
            "unit": "millions_eur"
        },
        "nominal_household_consumption": {
            "base_value": 76500,
            "growth_rate": 0.007, 
            "volatility": 0.008,
            "type": "nominal_gdp_component",
            "unit": "millions_eur"
        },
        "nominal_government_consumption": {
            "base_value": 25500,
            "growth_rate": 0.006, 
            "volatility": 0.006,
            "type": "nominal_gdp_component",
            "unit": "millions_eur"
        },
        "nominal_capitalformation": {
            "base_value": 28600,
            "growth_rate": 0.009, 
            "volatility": 0.017,
            "type": "nominal_gdp_component",
            "unit": "millions_eur"
        },
        "nominal_exports": {
            "base_value": 68300,
            "growth_rate": 0.010, 
            "volatility": 0.014,
            "type": "nominal_gdp_component",
            "unit": "millions_eur"
        },
        "nominal_imports": {
            "base_value": 62200,
            "growth_rate": 0.009, 
            "volatility": 0.012,
            "type": "nominal_gdp_component",
            "unit": "millions_eur"
        },
        "operating_surplus": {
            "base_value": 45000,
            "growth_rate": 0.004, 
            "volatility": 0.012,
            "type": "income",
            "unit": "millions_eur"
        },
        "compensation_employees": {
            "base_value": 75000,
            "growth_rate": 0.003, 
            "volatility": 0.005,
            "type": "income",
            "unit": "millions_eur"
        },
        "taxes_production": {
            "base_value": 15000,
            "growth_rate": 0.002, 
            "volatility": 0.008,
            "type": "tax",
            "unit": "millions_eur"
        },
        "real_gdp_ea": {
            "base_value": 9500000,
            "growth_rate": 0.002, 
            "volatility": 0.006,
            "type": "gdp_component",
            "unit": "millions_eur"
        }
    }


def generate_synthetic_target_data(variable_names: List[str], T: int = 20, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic target data for specified variables.

    Parameters:
    -----------
    variable_names : List[str]
        Names of variables to generate
    T : int
        Number of time periods
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Array of shape (T, len(variable_names)) containing synthetic time series data
    """
    np.random.seed(seed)
    
    metadata = get_variable_metadata()
    data = np.zeros((T, len(variable_names)))

    # Generate time series with trends and business cycle components
    for i, var_name in enumerate(variable_names):
        if var_name in metadata:
            var_meta = metadata[var_name]
            base = var_meta["base_value"]
            trend = var_meta["growth_rate"]
            vol = var_meta["volatility"]
        else:
            # Default values for unknown variables
            print(f"Warning: No metadata for variable '{var_name}', using defaults")
            base = 1000.0  # Default base value
            trend = 0.002  # Default growth rate
            vol = 0.01     # Default volatility

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
                cycle_component = ar_coeff * (
                    data[t - 1, i] - base * (1 + trend) ** (t - 1)
                ) + np.random.normal(0, vol * base)

            data[t, i] = trend_component + cycle_component

            # Ensure positive values for most variables (except rates/growth rates)
            if var_name not in ["euribor", "gdp_deflator_growth_ea"] and not var_name.endswith("_growth"):
                data[t, i] = max(data[t, i], base * 0.5)  # Minimum bound

    return data


def load_real_uk_data(variable_names: List[str], start_date: str = "2010Q1", periods: int = 20) -> np.ndarray:
    """
    Load real UK economic data for calibration.

    TODO: Implement loading real UK data from ONS or other sources.
    For now, returns synthetic data.

    Parameters:
    -----------
    variable_names : List[str]
        Names of variables to load
    start_date : str
        Start date in YYYY-QX format
    periods : int
        Number of quarters to load

    Returns:
    --------
    np.ndarray
        Array of shape (periods, len(variable_names)) containing real time series data
    """
    raise NotImplementedError("Real UK data loading not implemented")


def plot_target_data(data: np.ndarray, variable_names: List[str], save_path: str = None):
    """
    Plot the target data for visualization.

    Parameters:
    -----------
    data : np.ndarray
        Target data array of shape (T, n_variables)
    variable_names : List[str]
        Names of the variables
    save_path : str, optional
        Path to save the plot
    """
    T = data.shape[0]
    n_vars = len(variable_names)
    
    # Dynamic subplot layout based on number of variables
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle("Target Data for BeforeIT Model Calibration", fontsize=16)
    
    # Handle case where we have only one subplot
    if n_vars == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_vars == 1 else axes
    else:
        axes = axes.flat

    for i, (var, ax) in enumerate(zip(variable_names, axes)):
        ax.plot(range(T), data[:, i], "b-", linewidth=2)
        ax.set_title(var.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Time Period")
        ax.grid(True, alpha=0.3)

        # Special formatting for different variable types
        if "euribor" in var or "rate" in var:
            ax.set_ylabel("Rate")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3%}"))
        elif "deflator" in var or "index" in var:
            ax.set_ylabel("Index")
        elif "growth" in var:
            ax.set_ylabel("Growth Rate")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2%}"))
        else:
            ax.set_ylabel("Millions EUR")

    # Hide empty subplots
    for j in range(len(variable_names), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def get_calibration_targets(
    variable_names: List[str],
    data_source: str = "synthetic", 
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Get calibration target data and metadata.

    Parameters:
    -----------
    variable_names : List[str]
        Names of variables to generate/load
    data_source : str
        Data source: "synthetic" or "uk_real"
    **kwargs
        Additional arguments passed to data loading functions

    Returns:
    --------
    tuple
        (target_data, metadata) where target_data is shape (T, len(variable_names)) and
        metadata contains information about the data
    """
    if data_source == "synthetic":
        T = kwargs.get("T", 20)
        seed = kwargs.get("seed", 42)
        data = generate_synthetic_target_data(variable_names, T, seed)
        metadata = {
            "source": "synthetic",
            "periods": T,
            "seed": seed,
            "variables": variable_names,
            "description": "Synthetic data for BeforeIT model calibration",
        }
    elif data_source == "uk_real":
        start_date = kwargs.get("start_date", "2010Q1")
        periods = kwargs.get("periods", 20)
        data = load_real_uk_data(variable_names, start_date, periods)
        metadata = {
            "source": "uk_real",
            "start_date": start_date,
            "periods": periods,
            "variables": variable_names,
            "description": "Real UK economic data",
        }
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    return data, metadata


if __name__ == "__main__":
    # Example usage with default variables
    print("Generating synthetic target data...")
    
    # Use some common economic variables for demonstration
    default_variables = [
        "real_gdp",
        "real_household_consumption", 
        "real_government_consumption",
        "real_capitalformation",
        "real_exports",
        "real_imports",
        "wages",
        "euribor",
        "gdp_deflator_real_ea"
    ]
    
    target_data, metadata = get_calibration_targets(
        variable_names=default_variables,
        data_source="synthetic", 
        T=40, 
        seed=42
    )

    print(f"Target data shape: {target_data.shape}")
    print(f"Variables: {metadata['variables']}")

    # Plot the data
    plot_target_data(target_data, default_variables, "plots/target_data_plot.png")

    # Save data for calibration
    np.save("data/target_data.npy", target_data)
    print("Target data saved to target_data.npy")
