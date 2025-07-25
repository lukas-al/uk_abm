"""
Main calibration script for the BeforeIT model using black-it.

This script sets up and runs the calibration process for the BeforeIT economic model
using the black-it calibration toolbox directly.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple

# black-it imports
from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.samplers.best_batch import BestBatchSampler
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.r_sequence import RSequenceSampler
from black_it.samplers.random_forest import RandomForestSampler
from black_it.samplers.random_uniform import RandomUniformSampler
from black_it.samplers.gaussian_process import GaussianProcessSampler

# Local imports
from uk_abm.data_loader import get_calibration_targets, plot_target_data
from uk_abm.julia_model_wrapper import BeforeITModelWrapper

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_samplers(batch_size: int = 4) -> List:
    """
    Set up the sampling algorithms for calibration.

    Parameters:
    -----------
    batch_size : int
        Number of parameter sets to evaluate per iteration

    Returns:
    --------
    List
        List of configured samplers
    """
    logger.info(f"Setting up samplers with batch size {batch_size}")

    samplers = [
        RandomUniformSampler(batch_size=batch_size),
        RSequenceSampler(batch_size=batch_size),
        HaltonSampler(batch_size=batch_size),
        RandomForestSampler(batch_size=batch_size),
        BestBatchSampler(batch_size=batch_size),
        GaussianProcessSampler(batch_size=batch_size),
    ]

    return samplers


def run_calibration(
    model_wrapper: BeforeITModelWrapper,
    target_data: np.ndarray,
    results_folder: str,
    samplers: List,
    max_iterations: int = 20,
    ensemble_size: int = 4,
    loss_function: str = "msm",
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Calibrator]:
    """
    Run the calibration process using black-it Calibrator directly.

    Parameters:
    -----------
    model_wrapper : BeforeITModelWrapper
        The model wrapper instance
    target_data : np.ndarray
        Target data for calibration
    results_folder : str
        Folder to save calibration results
    samplers : List
        List of configured samplers to use for calibration
    max_iterations : int
        Maximum number of calibration iterations
    ensemble_size : int
        Number of ensemble runs per parameter set
    loss_function : str
        Loss function to use ("msm" for Method of Simulated Moments)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, Calibrator]
        (best_parameters, best_losses, calibrator_instance)
    """
    logger.info("Starting calibration process")

    # Set up loss function
    if loss_function == "msm":
        loss_func = MethodOfMomentsLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

    # Create model function with ensemble averaging
    def model_func(theta, N, rndSeed):
        results = model_wrapper.run_simulation(theta, N, rndSeed, ensemble_size)
        return results.to_calibration_array()

    # Get parameter information
    parameter_bounds = model_wrapper.get_parameter_bounds()
    parameter_precisions = model_wrapper.get_parameter_precisions()
    parameter_names = model_wrapper.get_parameter_names()

    # Initialize calibrator
    calibrator = Calibrator(
        samplers=samplers,
        real_data=target_data,
        model=model_func,
        parameters_bounds=parameter_bounds,
        parameters_precision=parameter_precisions,
        ensemble_size=1,  # We handle ensemble averaging in the wrapper
        loss_function=loss_func,
        saving_folder=results_folder,
        verbose=True,
        n_jobs=n_jobs,
    )

    # Run calibration
    logger.info(f"Running calibration for {max_iterations} iterations")
    logger.info(f"Parameter bounds: {dict(zip(parameter_names, parameter_bounds))}")

    try:
        best_params, best_losses = calibrator.calibrate(max_iterations)

        logger.info("Calibration completed successfully")
        logger.info(f"Best parameters: {dict(zip(parameter_names, best_params[0]))}")
        logger.info(f"Best loss: {best_losses[0]}")

        return best_params, best_losses, calibrator

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise


def plot_calibration_progress(
    calibrator: Calibrator, parameter_names: List[str], results_folder: str
):
    """Plot the convergence of the calibration process."""
    logger.info("Plotting calibration progress")

    # Get losses per batch
    batch_nums = calibrator.batch_num_samp
    max_batch = int(max(batch_nums)) + 1

    losses_per_batch = [
        calibrator.losses_samp[batch_nums == i] for i in range(max_batch)
    ]
    mins_per_batch = np.array(
        [np.min(l) if len(l) > 0 else np.inf for l in losses_per_batch]
    )
    cummin_per_batch = [
        np.min(mins_per_batch[: i + 1]) for i in range(len(mins_per_batch))
    ]

    plt.figure(figsize=(12, 4))

    # Plot 1: Loss evolution
    plt.subplot(1, 2, 1)
    plt.scatter(batch_nums, calibrator.losses_samp, alpha=0.6, s=30)
    plt.plot(
        range(len(cummin_per_batch)),
        cummin_per_batch,
        "r-",
        linewidth=2,
        label="Best so far",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Calibration Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Parameter convergence (first few parameters)
    plt.subplot(1, 2, 2)
    for i in range(min(3, len(parameter_names))):
        plt.scatter(
            batch_nums,
            calibrator.params_samp[:, i],
            alpha=0.6,
            s=20,
            label=parameter_names[i],
        )
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    plt.title("Parameter Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_folder, "calibration_progress.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_parameter_exploration(
    calibrator: Calibrator, parameter_names: List[str], results_folder: str
):
    """Plot parameter space exploration."""
    logger.info("Plotting parameter space exploration")

    n_params = len(parameter_names)

    # Create scatter plot matrix for first 4 parameters
    plot_params = min(4, n_params)
    fig, axes = plt.subplots(plot_params, plot_params, figsize=(12, 12))
    fig.suptitle("Parameter Space Exploration", fontsize=16)

    for i in range(plot_params):
        for j in range(plot_params):
            ax = axes[i, j] if plot_params > 1 else axes

            if i == j:
                # Diagonal: histogram
                ax.hist(
                    calibrator.params_samp[:, i], bins=20, alpha=0.7, color="skyblue"
                )
                ax.set_title(parameter_names[i])
            else:
                # Off-diagonal: scatter plot
                scatter = ax.scatter(
                    calibrator.params_samp[:, j],
                    calibrator.params_samp[:, i],
                    c=calibrator.losses_samp,
                    cmap="viridis",
                    alpha=0.6,
                    s=30,
                )

                # Mark best point
                idxmin = np.argmin(calibrator.losses_samp)
                ax.scatter(
                    calibrator.params_samp[idxmin, j],
                    calibrator.params_samp[idxmin, i],
                    marker="x",
                    s=200,
                    color="red",
                    linewidth=3,
                )

            if i == plot_params - 1:
                ax.set_xlabel(parameter_names[j])
            if j == 0:
                ax.set_ylabel(parameter_names[i])

    # Add colorbar
    if plot_params > 1:
        plt.colorbar(scatter, ax=axes, label="Loss", shrink=0.8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_folder, "parameter_exploration.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def compare_best_simulation(
    calibrator: Calibrator,
    model_wrapper: BeforeITModelWrapper,
    target_data: np.ndarray,
    results_folder: str,
):
    """Compare the best simulation with target data."""
    logger.info("Comparing best simulation with target data")

    # Get best parameters
    idxmin = np.argmin(calibrator.losses_samp)
    best_params = calibrator.params_samp[idxmin]

    # Run simulation with best parameters
    T = target_data.shape[0]
    simulation_results = model_wrapper.run_simulation(
        best_params, T, seed=42, ensemble_size=8
    )
    simulated_data = simulation_results.to_calibration_array()

    # Plot comparison
    variable_names = [
        "real_gdp",
        "real_household_consumption",
        "real_government_consumption",
        "real_capitalformation",
        "real_exports",
        "real_imports",
        "wages",
        "euribor",
        "gdp_deflator",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Target vs Best Simulation", fontsize=16)

    for i, (var, ax) in enumerate(zip(variable_names, axes.flat)):
        ax.plot(target_data[:, i], "b-", linewidth=2, label="Target", alpha=0.8)
        ax.plot(simulated_data[:, i], "r--", linewidth=2, label="Simulated", alpha=0.8)
        ax.set_title(var.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_folder, "target_vs_simulation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Save simulated data
    np.save(os.path.join(results_folder, "best_simulation.npy"), simulated_data)


def save_detailed_results(
    calibrator: Calibrator,
    model_wrapper: BeforeITModelWrapper,
    target_data: np.ndarray,
    data_metadata: Dict[str, Any],
    target_data_source: str,
    base_parameters: str,
    results_folder: str,
):
    """Save detailed calibration results."""
    logger.info("Saving detailed results")

    parameter_names = model_wrapper.get_parameter_names()
    parameter_bounds = model_wrapper.get_parameter_bounds()
    parameter_precisions = model_wrapper.get_parameter_precisions()

    results = {
        "parameter_names": parameter_names,
        "parameter_bounds": parameter_bounds,
        "parameter_precisions": parameter_precisions,
        "all_parameters": calibrator.params_samp,
        "all_losses": calibrator.losses_samp,
        "batch_numbers": calibrator.batch_num_samp,
        "target_data": target_data,
        "data_metadata": data_metadata,
        "best_loss": np.min(calibrator.losses_samp),
        "best_parameters": calibrator.params_samp[np.argmin(calibrator.losses_samp)],
    }

    np.save(os.path.join(results_folder, "calibration_results.npy"), results)

    # Also save as readable text
    with open(os.path.join(results_folder, "results_summary.txt"), "w") as f:
        f.write("BeforeIT Model Calibration Results\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Target data source: {target_data_source}\n")
        f.write(f"Base parameters: {base_parameters}\n")
        f.write(f"Target data shape: {target_data.shape}\n")
        f.write(f"Number of iterations: {len(np.unique(calibrator.batch_num_samp))}\n")
        f.write(f"Total evaluations: {len(calibrator.losses_samp)}\n\n")

        idxmin = np.argmin(calibrator.losses_samp)
        f.write(f"Best loss: {calibrator.losses_samp[idxmin]:.8f}\n\n")

        f.write("Best parameters:\n")
        for name, value in zip(parameter_names, calibrator.params_samp[idxmin]):
            f.write(f"  {name:15s}: {value:.6f}\n")


def analyze_calibration_results(
    calibrator: Calibrator,
    model_wrapper: BeforeITModelWrapper,
    target_data: np.ndarray,
    data_metadata: Dict[str, Any],
    target_data_source: str,
    base_parameters: str,
    results_folder: str,
):
    """
    Analyze and visualize calibration results.
    """
    logger.info("Analyzing calibration results")

    parameter_names = model_wrapper.get_parameter_names()

    # Find best parameters
    idxmin = np.argmin(calibrator.losses_samp)
    best_params = calibrator.params_samp[idxmin]
    best_loss = calibrator.losses_samp[idxmin]

    logger.info(f"Best parameter set (loss={best_loss:.6f}):")
    for name, value in zip(parameter_names, best_params):
        logger.info(f"  {name}: {value:.6f}")

    # Save best parameters
    best_params_dict = dict(zip(parameter_names, best_params))
    np.save(os.path.join(results_folder, "best_parameters.npy"), best_params_dict)

    # Generate all plots and analysis
    plot_calibration_progress(calibrator, parameter_names, results_folder)
    plot_parameter_exploration(calibrator, parameter_names, results_folder)
    compare_best_simulation(calibrator, model_wrapper, target_data, results_folder)
    save_detailed_results(
        calibrator,
        model_wrapper,
        target_data,
        data_metadata,
        target_data_source,
        base_parameters,
        results_folder,
    )


def main():
    """Main function to run the calibration using black-it Calibrator directly."""
    logger.info("Starting BeforeIT model calibration")

    # Configuration
    config = {
        "target_data_source": "synthetic",  # or 'uk_real'
        "base_parameters": "AUSTRIA2010Q1",
        "max_iterations": 15,
        "ensemble_size": 4,
        "batch_size": 6,
        "T": 20,  # Number of time periods
        "seed": 42,
    }

    # Create results folder
    results_folder = f"calibration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_folder, exist_ok=True)

    try:
        # Load target data
        logger.info(f"Loading target data from {config['target_data_source']}")
        target_data, data_metadata = get_calibration_targets(
            config["target_data_source"], T=config["T"], seed=config["seed"]
        )

        # Save target data plot
        plot_target_data(target_data, os.path.join(results_folder, "target_data.png"))

        # Initialize model wrapper
        logger.info("Initializing BeforeIT model wrapper")
        model_wrapper = BeforeITModelWrapper(base_parameters=config["base_parameters"])

        parameter_names = model_wrapper.get_parameter_names()
        logger.info(f"Calibrating {len(parameter_names)} parameters: {parameter_names}")
        logger.info(f"Target data shape: {target_data.shape}")

        # Set up samplers
        samplers = setup_samplers(config["batch_size"])

        # Run calibration using black-it Calibrator directly
        best_params, best_losses, calibrator = run_calibration(
            model_wrapper=model_wrapper,
            target_data=target_data,
            results_folder=results_folder,
            samplers=samplers,
            max_iterations=config["max_iterations"],
            ensemble_size=config["ensemble_size"],
        )

        # Analyze results
        analyze_calibration_results(
            calibrator=calibrator,
            model_wrapper=model_wrapper,
            target_data=target_data,
            data_metadata=data_metadata,
            target_data_source=config["target_data_source"],
            base_parameters=config["base_parameters"],
            results_folder=results_folder,
        )

        logger.info(f"Calibration completed. Results saved in: {results_folder}")

    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise


if __name__ == "__main__":
    main()
