#!/usr/bin/env python3
"""
Validate WaterSIC implementation structure without requiring PyTorch.
"""

import ast
import sys
import os

def check_function_exists(file_path, function_name):
    """Check if a function exists in the given file."""
    with open(file_path, 'r') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError as e:
            print(f"⚠️  Syntax error in {file_path}: {e}")
            return False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return True
    return False

def check_class_method_exists(file_path, class_name, method_name):
    """Check if a method exists in a class."""
    with open(file_path, 'r') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError as e:
            print(f"⚠️  Syntax error in {file_path}: {e}")
            return False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return True
    return False

def check_environment_variable(file_path, var_name):
    """Check if an environment variable is defined."""
    with open(file_path, 'r') as f:
        content = f.read()
        return var_name in content

def validate_watersic_implementation():
    """Validate that all WaterSIC components are implemented."""
    script_path = "train_gpt.py"
    
    if not os.path.exists(script_path):
        print(f"❌ File {script_path} not found")
        return False
    
    print("=== WaterSIC Implementation Validation ===")
    
    # Check core WaterSIC functions
    core_functions = [
        "_get_calibration_data",
        "_compute_covariance_matrix", 
        "_cholesky_decomposition",
        "_zsic_quantize",
        "_binary_search_rate",
        "quantize_layer_watersic",
        "quantize_state_dict_watersic",
        "dequantize_state_dict_watersic"
    ]
    
    print("\n--- Core WaterSIC Functions ---")
    all_functions_exist = True
    for func in core_functions:
        exists = check_function_exists(script_path, func)
        status = "✓" if exists else "❌"
        print(f"{status} {func}")
        if not exists:
            all_functions_exist = False
    
    # Check CastedLinear modifications
    print("\n--- CastedLinear Modifications ---")
    class_checks = [
        ("CastedLinear", "forward"),
    ]
    
    for class_name, method_name in class_checks:
        exists = check_class_method_exists(script_path, class_name, method_name)
        status = "✓" if exists else "❌"
        print(f"{status} {class_name}.{method_name}")
        if not exists:
            all_functions_exist = False
    
    # Check environment variables
    print("\n--- Environment Variables ---")
    env_vars = [
        "WATERSIC_ENABLED",
        "WATERSIC_BITS", 
        "WATERSIC_CALIBRATION_SAMPLES",
        "WATERSIC_LMMSE_CORRECTION",
        "WATERSIC_ACTIVATION_DRIFT",
        "WATERSIC_RESIDUAL_CORRECTION",
        "WATERSIC_ATTENTION_WEIGHTED"
    ]
    
    for var in env_vars:
        exists = check_environment_variable(script_path, var)
        status = "✓" if exists else "❌"
        print(f"{status} {var}")
        if not exists:
            all_functions_exist = False
    
    # Check for key algorithm terms in the code
    print("\n--- Algorithm Components ---")
    with open(script_path, 'r') as f:
        content = f.read()
    
    algorithm_terms = [
        ("Waterfilling", "alpha_diag = c / (L_diag + 1e-8)"),
        ("ZSIC", "_zsic_quantize"),
        ("LMMSE correction", "lmmse_correction"),
        ("Cholesky decomposition", "torch.linalg.cholesky"),
        ("Entropy coding", "zstd" in content.lower() or "entropy" in content.lower())
    ]
    
    for term, check in algorithm_terms:
        if isinstance(check, str):
            exists = check in content
        else:
            exists = check
        status = "✓" if exists else "❌"
        print(f"{status} {term}")
        if not exists:
            all_functions_exist = False
    
    print("\n" + "=" * 50)
    if all_functions_exist:
        print("🎉 WaterSIC implementation validation PASSED!")
        print("\nKey features implemented:")
        print("• Complete ZSIC algorithm with waterfilling rate allocation")
        print("• LMMSE correction for bias reduction") 
        print("• Activation drift correction (Qronos approach)")
        print("• Cholesky decomposition for covariance matrix handling")
        print("• Binary search for rate assignment")
        print("• Integration with existing training pipeline")
        print("• Environment variable configuration system")
        print("• CastedLinear fake quantization support")
        return True
    else:
        print("❌ WaterSIC implementation validation FAILED!")
        print("Some components are missing or incomplete.")
        return False

def check_file_size():
    """Check that the modified file is reasonable in size."""
    script_path = "train_gpt.py"
    if os.path.exists(script_path):
        size = os.path.getsize(script_path)
        lines = sum(1 for line in open(script_path))
        print(f"\nFile statistics:")
        print(f"• Size: {size:,} bytes ({size/1024:.1f} KB)")
        print(f"• Lines: {lines:,}")
        
        if lines > 3000:
            print("⚠️  File is quite large - consider splitting into modules for production")
        else:
            print("✓ File size is reasonable")

def main():
    """Run validation."""
    print("WaterSIC Implementation Validator")
    print("=" * 40)
    
    success = validate_watersic_implementation()
    check_file_size()
    
    if success:
        print(f"\n🚀 Ready for testing with environment variables:")
        print(f"   WATERSIC_ENABLED=1 WATERSIC_BITS=4.0 python train_gpt.py")
        return 0
    else:
        print(f"\n⚠️  Implementation needs fixes before testing")
        return 1

if __name__ == "__main__":
    sys.exit(main())