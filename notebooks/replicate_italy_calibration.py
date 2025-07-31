import marimo

__generated_with = "0.14.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
    from model_wrapper.julia_model_wrapper import BeforeITModelWrapper
    import black_it
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    return BeforeITModelWrapper, mo, np, pd, plt


@app.cell
def _():
    # import juliacall as jl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Replicate Italy Calibration

    The original ABM is calibrated on Italian data. I want to calibrate it on UK data, but I need to understand what parameters one should calibrate, estimate, etc. 

    So to do that, we're going to do the following:

    1. use the BeforeIT calibration to calibrate the model as in their tutorial.
    2. do the same with the BlackIT package and julia wrapper I've developed
    3. Then figure out what parameters we can pin, what data we need to collect for the UK calbration
    4. Then calibrate/estimate the model - exactly as it is - on the UK data!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Implement the BeforeIT calibration method""")
    return


@app.cell
def _(BeforeITModelWrapper):
    # We can use our existing wrapper to replicate their baseline calibration / estimation (WHICH IS IT????)
    model = BeforeITModelWrapper(
        base_parameters="ITALY2010Q1",
        model_path="dev/BeforeIT.jl",
        julia_threads=1,
    )
    return (model,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Available parameters
    Just printing out the parameters - includes the table from the original paper
    """
    )
    return


@app.cell
def _(model):
    # Let's check the data which they pass for calibration:
    jl_calibrator = model.jl.Bit.ITALY_CALIBRATION
    jl_calibrator

    # println(keys(cal.calibration))
    # println(keys(cal.figaro))
    # println(keys(cal.data))
    # println(keys(cal.ea))
    return (jl_calibrator,)


@app.cell
def _(jl_calibrator, np):
    for _key in jl_calibrator.calibration:
        print(_key)
        print("   ", type(jl_calibrator.calibration[_key]))
        print("   ", np.shape(jl_calibrator.calibration[_key]))
        print("   ", np.array(jl_calibrator.calibration[_key])[:5])
        print("------------------------------------------------")
    return


@app.cell
def _(jl_calibrator, np):
    # jl_calibrator.figaro
    for _key in jl_calibrator.figaro:
        print(_key)
        print("   ", type(jl_calibrator.figaro[_key]))
        print("   ", np.shape(jl_calibrator.figaro[_key]))
        print("   ", np.array(jl_calibrator.figaro[_key])[:5])
        print("------------------------------------------------")
    return


@app.cell
def _(jl_calibrator, np):
    for _key in jl_calibrator.data:
        print(_key)
        print("   ", type(jl_calibrator.data[_key]))
        print("   ", np.shape(jl_calibrator.data[_key]))
        print("   ", np.array(jl_calibrator.data[_key])[:5])
        print("------------------------------------------------")
    return


@app.cell
def _(jl_calibrator, np):
    for _key in jl_calibrator.ea:
        print(_key)
        print("   ", type(jl_calibrator.ea[_key]))
        print("   ", np.shape(jl_calibrator.ea[_key]))
        print("   ", np.array(jl_calibrator.ea[_key])[:5])
        print("------------------------------------------------")
    return


@app.cell
def _(jl_calibrator):
    # Maximum calibration date -> this is the last date to which we can calibrate the data?
    print(jl_calibrator.max_calibration_date)

    # Estimation date -> This is the date on which these parameters were estimated?
    print(jl_calibrator.estimation_date)

    return


@app.cell
def _(mo):
    mo.image("notebooks/public/model_parameters.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Run the before it calibration approach""")
    return


@app.cell
def _(jl_calibrator, model, pd):
    # Replicate the BeforeIT prediction pipeline approach
    # Using the calibration date from the ITALY_CALIBRATION object
    # The date needs to be end-of-quarter to match the available data.
    calibration_date = pd.Timestamp("2014-03-31")

    print(calibration_date)
    # Get parameters and initial conditions (equivalent to Bit.get_params_and_initial_conditions)
    parameters, initial_conditions = model.jl.Bit.get_params_and_initial_conditions(
        jl_calibrator, calibration_date, scale=0.0001
    )

    print(f"Calibration date: {calibration_date}")
    print(f"Number of parameters: {len(parameters)}")
    print(f"Number of initial conditions: {len(initial_conditions)}")

    return calibration_date, initial_conditions, parameters


@app.cell
def _(initial_conditions, model, parameters):
    # Run ensemble simulation (equivalent to Bit.ensemblerun)
    T = 20  # Number of quarters to simulate
    n_sims = 3  # Number of ensemble runs

    # Create model and run ensemble
    julia_model = model.jl.Bit.Model(parameters, initial_conditions)
    model_vector = model.jl.Bit.ensemblerun(julia_model, T, n_sims)

    print(f"Completed ensemble simulation: T={T}, n_sims={n_sims}")

    return (model_vector,)


@app.cell
def _(calibration_date, jl_calibrator, model, model_vector):
    # Get predictions from simulations (equivalent to Bit.get_predictions_from_sims)
    real_data = jl_calibrator.data
    predictions_dict = model.jl.Bit.get_predictions_from_sims(
        model.jl.Bit.DataVector(model_vector), real_data, calibration_date
    )

    print(f"Generated predictions for {len(predictions_dict)} variables")
    print("Available prediction variables:")
    for _i, key in enumerate(predictions_dict.keys()):
        if _i < 10:  # Show first 10 variables
            print(f"  - {key}")
        elif _i == 10:
            print(f"  ... and {len(predictions_dict) - 10} more")
            break

    return predictions_dict, real_data


@app.cell
def _(np, plt, predictions_dict, real_data):
    # Plot predictions vs real data (equivalent to Bit.plot_model_vs_real)

    # Key economic variables to plot
    variables_to_plot = [
        "real_gdp_quarterly",
        "real_household_consumption_quarterly", 
        "real_fixed_capitalformation_quarterly",
        "real_government_consumption_quarterly",
        "real_exports_quarterly",
        "real_imports_quarterly"
    ]

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for _i, var in enumerate(variables_to_plot):
        if var in predictions_dict and var in real_data:
            ax = axes[_i]

            # Plot real data
            real_values = np.array(real_data[var])
            ax.plot(real_values, 'b-', label='Real Data', linewidth=2)

            # Plot model predictions (mean across simulations)
            pred_values = np.array(predictions_dict[var])
            pred_mean = np.mean(pred_values, axis=1)
            ax.plot(pred_mean, 'r--', label='Model Prediction', linewidth=2)

            ax.set_title(var.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Redoing the estimation / calibration

    So what they've done, is they've calculated a bunch of numbers from:

    - Microdata (e.g. population, MPCs, etc.)
    - Aggregate data (e.g. taylor rule estimate)
    - National accounts.
    - and then fixed them.

    Some of these numbers vary through time, essentially re-instantiating the model each time step to perform the N step ahead forecast.

    For example:

    - population
    - nominal household consumption growth
    - etc.

    So to replicate this, we'd need to:

    1. Pick which parameters we're happy to use from Italy / Austria's calibration set. Use as many as possible here, to reduce the amount of things we need to estimate / calibrate ourselves.
    2. Find numbers for the data which we've decided we're going to fix and/or set a path for through time.
    3. Having set this, subsequently choose our outputs which we're going to simulate and compare against.
    """
    )
    return


if __name__ == "__main__":
    app.run()
