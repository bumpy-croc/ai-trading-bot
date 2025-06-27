#!/usr/bin/env python3
"""
Simple Model Validator for CI/CD Pipeline

This script validates models before deployment to catch issues early.
Use this in your CI/CD pipeline before deploying new models.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

def validate_model(model_path: str, expected_symbol: str = None) -> dict:
    """
    Validate a model file before deployment
    
    Returns:
        dict with validation results
    """
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        # Check 1: File exists and readable
        if not os.path.exists(model_path):
            results['valid'] = False
            results['errors'].append(f"Model file not found: {model_path}")
            return results
        
        # Check 2: File size reasonable
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        results['info']['file_size_mb'] = round(file_size, 2)
        
        if file_size < 0.1:
            results['valid'] = False
            results['errors'].append(f"Model file too small: {file_size:.2f}MB")
        elif file_size > 500:
            results['warnings'].append(f"Large model file: {file_size:.2f}MB")
        
        # Check 3: Can load with ONNX
        try:
            import onnx
            import onnxruntime as ort
            
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            session = ort.InferenceSession(model_path)
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            results['info']['input_shape'] = input_info.shape
            results['info']['input_name'] = input_info.name
            results['info']['output_shape'] = output_info.shape
            results['info']['output_name'] = output_info.name
            
            # Check 4: Test inference with dummy data
            if len(input_info.shape) >= 3:
                # Assume shape is [batch, sequence, features]
                batch_size = 1
                sequence_length = input_info.shape[1] if input_info.shape[1] else 120
                feature_count = input_info.shape[2] if input_info.shape[2] else 5
                
                dummy_input = np.random.random((batch_size, sequence_length, feature_count)).astype(np.float32)
                
                try:
                    output = session.run(None, {input_info.name: dummy_input})
                    results['info']['test_inference'] = 'success'
                    results['info']['output_value_range'] = [float(np.min(output[0])), float(np.max(output[0]))]
                except Exception as e:
                    results['valid'] = False
                    results['errors'].append(f"Test inference failed: {e}")
            
        except ImportError:
            results['warnings'].append("ONNX/ONNXRuntime not available - skipping model loading test")
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Model loading failed: {e}")
        
        # Check 5: Validate against expected symbol (if provided)
        if expected_symbol:
            model_filename = os.path.basename(model_path).lower()
            expected_filename_part = expected_symbol.lower()
            
            if expected_filename_part not in model_filename:
                results['warnings'].append(f"Model filename doesn't contain expected symbol: {expected_symbol}")
        
        # Check 6: Look for metadata file
        metadata_path = model_path.replace('.onnx', '_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    results['info']['metadata'] = metadata
            except Exception as e:
                results['warnings'].append(f"Could not read metadata file: {e}")
        else:
            results['warnings'].append("No metadata file found")
        
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Validation error: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Validate model for deployment')
    parser.add_argument('model_path', help='Path to model file')
    parser.add_argument('--symbol', help='Expected trading symbol')
    parser.add_argument('--strict', action='store_true', help='Treat warnings as errors')
    
    args = parser.parse_args()
    
    print(f"🔍 Validating model: {args.model_path}")
    
    results = validate_model(args.model_path, args.symbol)
    
    # Print results
    print(f"\n📊 Validation Results:")
    print(f"Valid: {'✅ Yes' if results['valid'] else '❌ No'}")
    
    if results['info']:
        print(f"\n📋 Model Info:")
        for key, value in results['info'].items():
            print(f"  {key}: {value}")
    
    if results['errors']:
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  • {error}")
    
    if results['warnings']:
        print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  • {warning}")
    
    # Exit with appropriate code
    if not results['valid']:
        print(f"\n❌ Model validation FAILED")
        sys.exit(1)
    elif args.strict and results['warnings']:
        print(f"\n❌ Model validation FAILED (strict mode, warnings treated as errors)")
        sys.exit(1)
    else:
        print(f"\n✅ Model validation PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main() 