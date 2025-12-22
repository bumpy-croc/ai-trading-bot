"""Tests for atb tests (legacy) commands."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cli.commands.tests import _collect_failures, _format_failures


class TestCollectFailures:
    """Tests for the _collect_failures function."""

    def test_collects_failures_from_junit_xml(self):
        """Test that failures are collected from JUnit XML."""
        # Arrange
        junit_xml = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
    <testsuite name="pytest">
        <testcase classname="test_example" name="test_pass"/>
        <testcase classname="test_example" name="test_fail">
            <failure message="AssertionError">
                Expected 2 but got 1
            </failure>
        </testcase>
    </testsuite>
</testsuites>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(junit_xml)
            xml_path = Path(f.name)

        try:
            # Act
            failures = _collect_failures(xml_path)

            # Assert
            assert len(failures) == 1
            assert failures[0]["classname"] == "test_example"
            assert failures[0]["name"] == "test_fail"
            assert "AssertionError" in failures[0]["message"]
        finally:
            xml_path.unlink()

    def test_returns_empty_list_for_nonexistent_file(self):
        """Test that empty list is returned for nonexistent file."""
        # Arrange
        xml_path = Path("/nonexistent/file.xml")

        # Act
        failures = _collect_failures(xml_path)

        # Assert
        assert len(failures) == 0


class TestFormatFailures:
    """Tests for the _format_failures function."""

    def test_formats_failures_correctly(self):
        """Test that failures are formatted correctly."""
        # Arrange
        failures = [
            {
                "classname": "test_example",
                "name": "test_fail",
                "message": "AssertionError",
                "details": "Expected 2 but got 1",
            }
        ]

        # Act
        output = _format_failures(failures, "Unit Tests")

        # Assert
        assert "test_example.test_fail" in output
        assert "AssertionError" in output
        assert "Expected 2 but got 1" in output

    def test_handles_empty_failures_list(self):
        """Test that empty failures list is handled."""
        # Arrange
        failures = []

        # Act
        output = _format_failures(failures, "Unit Tests")

        # Assert
        assert "No detailed failure information" in output
