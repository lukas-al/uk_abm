"""
Test script for Julia integration using JuliaCall.

This script verifies that JuliaCall is properly installed and the BeforeIT.jl package
is accessible from Python via the juliacall interface.
"""

import sys
import os


def test_juliacall_installation():
    """Test that juliacall is installed and accessible."""
    print("Testing JuliaCall installation...")

    try:
        import juliacall

        print("✓ JuliaCall package found")

        # Test basic Julia access
        jl = juliacall.newmodule("TestModule")
        result = jl.seval("1 + 1")
        if result == 2:
            print("✓ Basic Julia evaluation working")
            return True
        else:
            print(f"✗ Unexpected result from Julia: {result}")
            return False

    except ImportError:
        print("✗ JuliaCall not found. Install with: pip install juliacall")
        return False
    except Exception as e:
        print(f"✗ Error testing JuliaCall: {e}")
        return False


def test_beforeit_package():
    """Test that BeforeIT.jl package is accessible via JuliaCall."""
    print("\nTesting BeforeIT.jl package via JuliaCall...")

    try:
        import juliacall

        jl = juliacall.newmodule("BeforeITTest")

        # Find BeforeIT package path
        potential_paths = [
            os.path.join(os.getcwd(), "dev", "BeforeIT.jl"),
            os.path.join(os.getcwd(), "..", "..", "dev", "BeforeIT.jl"),
            os.path.join(os.path.dirname(os.getcwd()), "dev", "BeforeIT.jl"),
        ]

        beforeit_path = None
        for path in potential_paths:
            if os.path.isdir(path) and os.path.isfile(
                os.path.join(path, "src", "BeforeIT.jl")
            ):
                beforeit_path = path
                break

        if beforeit_path is None:
            print("✗ BeforeIT.jl package not found")
            return False

        print(f"Found BeforeIT.jl at: {beforeit_path}")

        # Activate BeforeIT environment
        jl.seval("using Pkg")
        jl.Pkg.activate(beforeit_path)

        # Install dependencies
        print("Installing package dependencies...")
        jl.Pkg.instantiate()

        # Import BeforeIT
        jl.seval("import BeforeIT as Bit")

        # Test basic functionality
        params = jl.Bit.AUSTRIA2010Q1.parameters
        initial_conditions = jl.Bit.AUSTRIA2010Q1.initial_conditions

        # Check that key parameters exist
        required_params = ["psi", "tau_INC", "mu"]
        for param in required_params:
            if param not in params:
                print(f"✗ Required parameter '{param}' not found")
                return False

        print("✓ BeforeIT.jl package accessible via JuliaCall")
        print(f"✓ Found required parameters: {required_params}")
        return True

    except Exception as e:
        print(f"✗ BeforeIT.jl package test failed: {e}")
        return False


def test_beforeit_simulation():
    """Test running a simple BeforeIT simulation via JuliaCall."""
    print("\nTesting BeforeIT simulation via JuliaCall...")

    try:
        import juliacall

        jl = juliacall.newmodule("BeforeITSimTest")

        # Find and activate BeforeIT
        potential_paths = [
            os.path.join(os.getcwd(), "dev", "BeforeIT.jl"),
            os.path.join(os.getcwd(), "..", "..", "dev", "BeforeIT.jl"),
            os.path.join(os.path.dirname(os.getcwd()), "dev", "BeforeIT.jl"),
        ]

        beforeit_path = None
        for path in potential_paths:
            if os.path.isdir(path) and os.path.isfile(
                os.path.join(path, "src", "BeforeIT.jl")
            ):
                beforeit_path = path
                break

        if beforeit_path is None:
            print("✗ BeforeIT.jl package not found")
            return False

        jl.seval("using Pkg")
        jl.Pkg.activate(beforeit_path)

        # Install dependencies
        print("Installing package dependencies...")
        jl.Pkg.instantiate()

        jl.seval("import BeforeIT as Bit")
        jl.seval("using Random")

        # Test basic simulation
        parameters = jl.Bit.AUSTRIA2010Q1.parameters
        initial_conditions = jl.Bit.AUSTRIA2010Q1.initial_conditions

        # Set random seed
        getattr(jl.Random, "seed!")(42)

        # Create model
        model = jl.Bit.Model(parameters, initial_conditions)

        # Run for 3 steps
        for t in range(3):
            getattr(jl.Bit, "step!")(model, multi_threading=False)
            getattr(jl.Bit, "update_data!")(model)

        # Check that we have data
        gdp_data = model.data.real_gdp
        import numpy as np

        gdp_array = np.array(gdp_data)

        if len(gdp_array) >= 3:
            print("✓ BeforeIT simulation successful via JuliaCall")
            print(f"✓ GDP data length: {len(gdp_array)}")
            print(f"✓ GDP values (first 3): {gdp_array[:3]}")
            return True
        else:
            print("✗ Insufficient data generated")
            return False

    except Exception as e:
        print(f"✗ BeforeIT simulation failed: {e}")
        return False


def test_parameter_modification():
    """Test modifying parameters and running simulation."""
    print("\nTesting parameter modification via JuliaCall...")

    try:
        import juliacall

        jl = juliacall.newmodule("BeforeITParamTest")

        # Find and activate BeforeIT
        potential_paths = [
            os.path.join(os.getcwd(), "dev", "BeforeIT.jl"),
            os.path.join(os.getcwd(), "..", "..", "dev", "BeforeIT.jl"),
            os.path.join(os.path.dirname(os.getcwd()), "dev", "BeforeIT.jl"),
        ]

        beforeit_path = None
        for path in potential_paths:
            if os.path.isdir(path) and os.path.isfile(
                os.path.join(path, "src", "BeforeIT.jl")
            ):
                beforeit_path = path
                break

        if beforeit_path is None:
            print("✗ BeforeIT.jl package not found")
            return False

        jl.seval("using Pkg")
        jl.Pkg.activate(beforeit_path)

        # Install dependencies
        print("Installing package dependencies...")
        jl.Pkg.instantiate()

        jl.seval("import BeforeIT as Bit")
        jl.seval("using Random")

        # Get base parameters and modify them
        base_params = jl.copy(jl.Bit.AUSTRIA2010Q1.parameters)
        initial_conditions = jl.Bit.AUSTRIA2010Q1.initial_conditions

        # Modify a parameter
        original_psi = float(base_params["psi"])
        modified_psi = 0.7  # Different from original
        base_params["psi"] = modified_psi

        print(
            f"✓ Modified parameter 'psi' from {original_psi:.3f} to {modified_psi:.3f}"
        )

        # Run simulation with modified parameters
        getattr(jl.Random, "seed!")(42)
        model = jl.Bit.Model(base_params, initial_conditions)

        # Run for 2 steps
        for t in range(2):
            getattr(jl.Bit, "step!")(model, multi_threading=False)
            getattr(jl.Bit, "update_data!")(model)

        # Verify simulation completed
        gdp_data = model.data.real_gdp
        import numpy as np

        gdp_array = np.array(gdp_data)

        if len(gdp_array) >= 2:
            print("✓ Parameter modification and simulation successful")
            return True
        else:
            print("✗ Simulation with modified parameters failed")
            return False

    except Exception as e:
        print(f"✗ Parameter modification test failed: {e}")
        return False


def test_python_wrapper():
    """Test the Python wrapper functionality."""
    print("\nTesting Python wrapper with JuliaCall...")

    try:
        from uk_abm.julia_model_wrapper import BeforeITModelWrapper

        # Initialize wrapper
        wrapper = BeforeITModelWrapper()

        # Test parameter methods
        bounds = wrapper.get_parameter_bounds()
        names = wrapper.get_parameter_names()
        precisions = wrapper.get_parameter_precisions()
        current_values = wrapper.get_current_parameter_values()
        variable_names = wrapper.get_variable_names()

        if len(bounds) == len(names) == len(precisions) == len(current_values):
            print(f"✓ Parameter methods working ({len(names)} parameters)")
        else:
            print("✗ Parameter method length mismatch")
            return False

        # Test with default parameters
        test_params = (bounds[0, :] + bounds[1, :]) / 2

        print("  Running test simulation...")
        result = wrapper.run_single_simulation(test_params, T=3, seed=42)
        
        # Convert to calibration array for testing
        calibration_array = result.to_calibration_array()
        
        print(f"  ModelResults contains {len(result.variables)} variables")
        print(f"  Time series variables: {result.list_timeseries_variables()}")
        print(f"  Panel variables: {result.list_panel_variables()}")
        print(f"  Calibration array shape: {calibration_array.shape}")

        if calibration_array.shape[0] == 3:  # Check time dimension
            print("✓ Python wrapper simulation successful")
            return True
        else:
            print(
                f"✗ Unexpected time dimension: {calibration_array.shape[0]}, expected 3"
            )
            return False

    except ImportError as e:
        print(f"✗ Cannot import wrapper: {e}")
        return False
    except Exception as e:
        print(f"✗ Wrapper test failed: {e}")
        return False


def test_ensemble_simulation():
    """Test ensemble simulation functionality."""
    print("\nTesting ensemble simulation...")

    try:
        from uk_abm.julia_model_wrapper import BeforeITModelWrapper

        wrapper = BeforeITModelWrapper()
        bounds = wrapper.get_parameter_bounds()
        test_params = (bounds[0, :] + bounds[1, :]) / 2

        print("  Running ensemble simulation...")
        result = wrapper.run_simulation(test_params, T=3, seed=42, ensemble_size=2)
        
        # Convert to calibration array for testing
        calibration_array = result.to_calibration_array(include_aggregated_panels=False)
        
        print(f"  Ensemble ModelResults contains {len(result.variables)} variables")
        print(f"  Calibration array shape: {calibration_array.shape}")

        if calibration_array.shape[0] == 3:  # Check time dimension
            print("✓ Ensemble simulation successful")
            return True
        else:
            print(
                f"✗ Unexpected ensemble time dimension: {calibration_array.shape[0]}, expected 3"
            )
            return False

    except Exception as e:
        print(f"✗ Ensemble simulation test failed: {e}")
        return False


def check_dependencies():
    """Check Python dependencies for the calibration system."""
    print("\nChecking Python dependencies...")

    required_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "black_it",
        "sklearn",
        "optuna",
        "juliacall",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == "sklearn":
                import sklearn
            elif package == "black_it":
                import black_it
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install juliacall black-it scikit-learn optuna")
        return False
    else:
        print("✓ All Python dependencies available")
        return True


def main():
    """Run all integration tests."""
    print("BeforeIT JuliaCall Integration Test")
    print("=" * 40)

    tests = [
        ("Python Dependencies", check_dependencies),
        ("JuliaCall Installation", test_juliacall_installation),
        ("BeforeIT Package", test_beforeit_package),
        ("BeforeIT Simulation", test_beforeit_simulation),
        ("Parameter Modification", test_parameter_modification),
        ("Python Wrapper", test_python_wrapper),
        ("Ensemble Simulation", test_ensemble_simulation),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Test: {test_name}")
        print("=" * 60)

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:25s}: {status}")
        if success:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! The system is ready for calibration.")
    else:
        print("\n❌ Some tests failed. Please address the issues above.")
        print("\nTroubleshooting tips:")
        print("- Install JuliaCall: pip install juliacall")
        print("- Check that dev/BeforeIT.jl directory exists")
        print("- Install missing Python packages: pip install -r requirements.txt")
        print("- Ensure Julia packages are properly installed")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
