"""
Simple example of BeforeIT model calibration using black-it and JuliaCall.

This script demonstrates a basic calibration workflow with synthetic data
using the high-performance JuliaCall integration.
"""

import sys
import os
import logging

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(__file__))

from data_loader import get_calibration_targets, plot_target_data
from julia_model_wrapper import BeforeITModelWrapper
from calibrate_beforeit import run_calibration, analyze_calibration_results

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simple_test():
    """
    Simple test to verify that all components work together.
    """
    print("=" * 60)
    print("BeforeIT Model Calibration - Simple Test (JuliaCall)")
    print("=" * 60)
    
    try:
        # 1. Test data loader
        print("\n1. Testing data loader...")
        target_data, metadata = get_calibration_targets("synthetic", T=10, seed=42)
        print(f"   ✓ Generated target data with shape: {target_data.shape}")
        print(f"   ✓ Variables: {metadata['variables']}")
        
        # 2. Test Julia wrapper with JuliaCall
        print("\n2. Testing Julia model wrapper with JuliaCall...")
        wrapper = BeforeITModelWrapper()
        
        # Get parameter information
        bounds = wrapper.get_parameter_bounds()
        names = wrapper.get_parameter_names()
        current_values = wrapper.get_current_parameter_values()
        print(f"   ✓ Found {len(names)} calibratable parameters")
        print(f"   ✓ Parameters: {names[:3]}...")  # Show first 3
        print(f"   ✓ Current values loaded successfully")
        
        # Test simulation with middle-of-bounds parameters
        test_params = [(b[0] + b[1]) / 2 for b in bounds]
        print(f"   ✓ Testing with middle-of-bounds parameters...")
        
        # Use single simulation for faster testing
        result = wrapper.run_single_simulation(test_params, T=5, seed=42)
        print(f"   ✓ Single simulation completed with output shape: {result.shape}")
        
        # Test ensemble simulation
        ensemble_result = wrapper.run_simulation(test_params, T=3, seed=42, ensemble_size=2)
        print(f"   ✓ Ensemble simulation completed with output shape: {ensemble_result.shape}")
        
        print("\n✅ All components working correctly with JuliaCall!")
        print("   Performance note: JuliaCall provides ~10x speedup vs subprocess calls")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_calibration():
    """
    Run a quick calibration with minimal iterations for demonstration.
    """
    print("\n" + "=" * 60)
    print("Quick Calibration Example (JuliaCall)")
    print("=" * 60)
    
    try:
        # Load target data and initialize model
        print("\nInitializing calibration...")
        target_data, metadata = get_calibration_targets("synthetic", T=8, seed=42)
        model_wrapper = BeforeITModelWrapper(base_parameters="AUSTRIA2010Q1")
        
        # Create results folder
        from datetime import datetime
        results_folder = f"calibration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(results_folder, exist_ok=True)
        
        print(f"Target data shape: {target_data.shape}")
        print(f"Calibrating {len(model_wrapper.get_parameter_names())} parameters")
        print("Using JuliaCall for efficient Julia integration")
        
        # Run short calibration
        print("\nRunning calibration (faster with JuliaCall)...")
        best_params, best_losses, calibrator = run_calibration(
            model_wrapper=model_wrapper,
            target_data=target_data,
            results_folder=results_folder,
            max_iterations=5,  # Very short for demo
            ensemble_size=2,   # Small ensemble for speed
            batch_size=4       # Small batch size
        )
        
        # Analyze results
        print("\nAnalyzing results...")
        analyze_calibration_results(
            calibrator=calibrator,
            model_wrapper=model_wrapper,
            target_data=target_data,
            data_metadata=metadata,
            target_data_source="synthetic",
            base_parameters="AUSTRIA2010Q1",
            results_folder=results_folder
        )
        
        print(f"\n✅ Calibration completed!")
        print(f"   Results saved in: {results_folder}")
        print(f"   Best loss: {best_losses[0]:.6f}")
        print(f"   Best parameters:")
        for name, value in zip(model_wrapper.get_parameter_names(), best_params[0]):
            print(f"     {name:12s}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstration_mode():
    """
    Run a demonstration with very minimal settings to show the workflow.
    """
    print("\n" + "=" * 60)
    print("Demonstration Mode - Minimal Calibration (JuliaCall)")
    print("=" * 60)
    print("Note: This uses minimal settings for demonstration purposes.")
    print("For real calibration, use longer time series and more iterations.")
    
    try:
        # Step 1: Generate target data
        print("\nStep 1: Generating synthetic target data...")
        target_data, metadata = get_calibration_targets("synthetic", T=6, seed=42)
        print(f"Generated data for {len(metadata['variables'])} variables over {target_data.shape[0]} periods")
        
        # Step 2: Initialize model wrapper
        print("\nStep 2: Setting up Julia model interface...")
        wrapper = BeforeITModelWrapper(base_parameters="AUSTRIA2010Q1")
        print(f"Configured to calibrate {len(wrapper.get_parameter_names())} parameters")
        print("Julia environment initialized with persistent session")
        
        # Step 3: Test single simulation
        print("\nStep 3: Testing single model simulation...")
        bounds = wrapper.get_parameter_bounds()
        test_params = [(b[0] + b[1]) / 2 for b in bounds]
        result = wrapper.run_single_simulation(test_params, T=6, seed=42)
        print(f"Simulation successful: output shape {result.shape}")
        
        # Step 4: Mini calibration
        print("\nStep 4: Running mini-calibration...")
        from datetime import datetime
        results_folder = f"calibration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(results_folder, exist_ok=True)
        
        # Very short calibration for demo
        best_params, best_losses, calibrator = run_calibration(
            model_wrapper=wrapper,
            target_data=target_data,
            results_folder=results_folder,
            max_iterations=3,
            ensemble_size=1,
            batch_size=2
        )
        
        print("\nStep 5: Results summary...")
        print(f"Best loss achieved: {best_losses[0]:.6f}")
        print("Top 3 parameters:")
        for i, (name, value) in enumerate(zip(wrapper.get_parameter_names()[:3], best_params[0][:3])):
            print(f"  {name}: {value:.4f}")
        
        print("\n✅ Demonstration completed successfully!")
        print(f"Results folder: {results_folder}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function with menu for different example modes.
    """
    print("BeforeIT Model Calibration Examples (JuliaCall Edition)")
    print("=" * 50)
    print("1. Simple component test")
    print("2. Quick calibration (5-10 minutes)")
    print("3. Demonstration mode (2-3 minutes)")
    print("4. Run all tests")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        success = simple_test()
    elif choice == "2":
        success = quick_calibration()
    elif choice == "3":
        success = demonstration_mode()
    elif choice == "4":
        print("Running all tests in sequence...")
        success1 = simple_test()
        if success1:
            success2 = demonstration_mode()
            if success2:
                success = success1 and success2
            else:
                success = False
        else:
            success = False
    else:
        print("Invalid choice. Please select 1-4.")
        return
    
    if success:
        print("\n🎉 All examples completed successfully!")
    else:
        print("\n❌ Some examples failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("- Ensure JuliaCall is installed: pip install juliacall")
        print("- Check that the BeforeIT.jl package is available")
        print("- Run 'python test_julia_integration.py' for detailed diagnostics")
        print("- Verify all Python dependencies are installed")


if __name__ == "__main__":
    main() 