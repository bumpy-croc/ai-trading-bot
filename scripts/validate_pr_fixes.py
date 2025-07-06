#!/usr/bin/env python3
"""
Validation script for PR review fixes
Verifies that log_event parameter binding issues are resolved
"""

import sys
import os
import re
from pathlib import Path

def check_log_event_calls():
    """Check that all log_event calls use keyword arguments properly"""
    
    print("🔍 Validating log_event Call Fixes")
    print("=" * 50)
    
    # Files to check
    files_to_check = [
        "src/database/manager.py",
        "src/live/trading_engine.py", 
        "scripts/railway_database_setup.py",
        "scripts/verify_database_connection.py",
        "scripts/test_database.py"
    ]
    
    issues_found = []
    fixes_verified = []
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
            
        print(f"\n📄 Checking: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find all log_event calls
        log_event_pattern = r'\.log_event\s*\('
        matches = re.finditer(log_event_pattern, content)
        
        for match in matches:
            # Extract the call context
            start = match.start()
            # Find the closing parenthesis
            paren_count = 0
            pos = start
            while pos < len(content):
                if content[pos] == '(':
                    paren_count += 1
                elif content[pos] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        break
                pos += 1
            
            call_text = content[start:pos+1]
            
            # Check if it uses keyword arguments
            if 'event_type=' in call_text or 'message=' in call_text:
                fixes_verified.append(f"  ✅ {file_path}: Uses keyword arguments")
                print(f"    ✅ Found proper keyword argument usage")
            else:
                # Check if it's a positional call that could have issues
                if 'details=' not in call_text and 'details' in call_text:
                    issues_found.append(f"  ❌ {file_path}: Potential positional argument issue")
                    print(f"    ❌ Potential issue with positional arguments")
                else:
                    print(f"    ✅ Call appears correct")
    
    return issues_found, fixes_verified


def check_method_signatures():
    """Check that new methods exist in DatabaseManager"""
    
    print(f"\n🔍 Validating New Method Signatures")
    print("-" * 40)
    
    db_manager_path = "src/database/manager.py"
    
    if not os.path.exists(db_manager_path):
        print(f"❌ Database manager not found: {db_manager_path}")
        return False
    
    with open(db_manager_path, 'r') as f:
        content = f.read()
    
    # Methods to check
    methods_to_check = [
        "test_connection",
        "get_database_info", 
        "get_connection_stats",
        "cleanup_connection_pool"
    ]
    
    methods_found = []
    methods_missing = []
    
    for method in methods_to_check:
        if f"def {method}(" in content:
            methods_found.append(method)
            print(f"  ✅ {method}() - Found")
        else:
            methods_missing.append(method)
            print(f"  ❌ {method}() - Missing")
    
    return len(methods_missing) == 0, methods_found, methods_missing


def check_test_files():
    """Check that test files exist"""
    
    print(f"\n🔍 Validating Test Files")
    print("-" * 30)
    
    test_files = [
        "tests/test_database_new_methods.py",
        "scripts/run_database_tests.py"
    ]
    
    files_found = []
    files_missing = []
    
    for file_path in test_files:
        if os.path.exists(file_path):
            files_found.append(file_path)
            print(f"  ✅ {file_path} - Found")
        else:
            files_missing.append(file_path)
            print(f"  ❌ {file_path} - Missing")
    
    return len(files_missing) == 0, files_found, files_missing


def main():
    """Main validation function"""
    
    print("🔧 PR Review Fixes Validation")
    print("=" * 60)
    print()
    print("Checking fixes for:")
    print("  1. Unit tests for new methods")
    print("  2. Parameter ordering fix for log_event")
    print("  3. Keyword arguments for log_event calls")
    print()
    
    # Check log_event calls
    issues_found, fixes_verified = check_log_event_calls()
    
    # Check method signatures
    methods_ok, methods_found, methods_missing = check_method_signatures()
    
    # Check test files
    tests_ok, tests_found, tests_missing = check_test_files()
    
    # Summary
    print(f"\n📊 Validation Summary")
    print("=" * 30)
    
    print(f"\n1. Parameter Binding Fixes:")
    if issues_found:
        print("  ❌ Issues found:")
        for issue in issues_found:
            print(issue)
    else:
        print("  ✅ No parameter binding issues detected")
    
    print(f"\n2. New Methods:")
    if methods_ok:
        print("  ✅ All new methods found")
        for method in methods_found:
            print(f"    - {method}()")
    else:
        print("  ❌ Missing methods:")
        for method in methods_missing:
            print(f"    - {method}()")
    
    print(f"\n3. Test Files:")
    if tests_ok:
        print("  ✅ All test files found")
        for test_file in tests_found:
            print(f"    - {test_file}")
    else:
        print("  ❌ Missing test files:")
        for test_file in tests_missing:
            print(f"    - {test_file}")
    
    print(f"\n4. Keyword Arguments:")
    if fixes_verified:
        print("  ✅ Keyword argument fixes verified:")
        for fix in fixes_verified:
            print(fix)
    else:
        print("  ⚠️  No keyword argument usage detected")
    
    # Overall result
    all_ok = not issues_found and methods_ok and tests_ok
    
    print(f"\n{'='*60}")
    if all_ok:
        print("🎉 All PR review fixes validated successfully!")
        print("✅ Ready for merge")
    else:
        print("❌ Some issues found - please review")
        print("❗ Not ready for merge")
    
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)