#!/usr/bin/env python3
"""
Parse JUnit XML files and extract detailed failure information.

This script processes JUnit XML test reports and outputs detailed failure
information including test names, error messages, and stack traces.

Usage:
    python3 scripts/parse_junit_failures.py <xml_file_path>

Args:
    xml_file_path: Path to the JUnit XML file to parse

Output:
    Detailed failure information printed to stdout
"""

import sys
import xml.etree.ElementTree as ET


def parse_junit_xml(xml_file_path: str) -> list[dict[str, str]]:
    """
    Parse JUnit XML file and extract failure information.
    
    Args:
        xml_file_path: Path to the JUnit XML file
        
    Returns:
        List of dictionaries containing failure details
    """
    failures = []
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        for testcase in root.findall('.//testcase'):
            failure = testcase.find('failure')
            if failure is not None:
                classname = testcase.get('classname', 'Unknown')
                name = testcase.get('name', 'Unknown')
                message = failure.get('message', 'No message')
                details = failure.text or 'No details'
                
                failures.append({
                    'classname': classname,
                    'name': name,
                    'message': message,
                    'details': details
                })
                
    except Exception as e:
        print(f'Error parsing XML file {xml_file_path}: {e}', file=sys.stderr)
        return []
    
    return failures


def format_failure_output(failures: list[dict[str, str]], test_type: str = "Unknown") -> str:
    """
    Format failure information for output.
    
    Args:
        failures: List of failure dictionaries
        test_type: Type of test (e.g., "Unit Tests", "Integration Tests")
        
    Returns:
        Formatted string with failure details
    """
    if not failures:
        return f'No detailed failure information found in {test_type} XML'
    
    output_lines = []
    for failure in failures:
        output_lines.append(f'Test: {failure["classname"]}.{failure["name"]}')
        output_lines.append(f'Message: {failure["message"]}')
        
        # Truncate details if too long
        details = failure["details"]
        if len(details) > 500:
            details = details[:500] + "..."
        output_lines.append(f'Details: {details}')
        output_lines.append('---')
    
    return '\n'.join(output_lines)


def main():
    """Main function to parse JUnit XML and output failure details."""
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/parse_junit_failures.py <xml_file_path>", file=sys.stderr)
        sys.exit(1)
    
    xml_file_path = sys.argv[1]
    
    # Parse the XML file
    failures = parse_junit_xml(xml_file_path)
    
    # Format and output the results
    output = format_failure_output(failures)
    print(output)


if __name__ == "__main__":
    main()