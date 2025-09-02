"""
Integration tests for Railway log analysis system.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.monitoring.log_analyzer import RailwayLogAnalyzer
from src.monitoring.railway_log_fetcher import RailwayLogFetcher


@pytest.mark.integration
class TestRailwayLogSystem:
    """Integration tests for the complete Railway log analysis system."""

    def test_log_analysis_pipeline(self):
        """Test the complete log analysis pipeline with mock data."""
        # * Create sample log content
        sample_logs = "\n".join([
            '{"timestamp": "2025-01-09T12:00:00Z", "level": "ERROR", "logger": "atb.trading", "message": "API rate limit exceeded", "component": "binance_provider"}',
            '{"timestamp": "2025-01-09T12:01:00Z", "level": "WARNING", "logger": "atb.risk", "message": "High volatility detected", "component": "risk_manager"}',
            '{"timestamp": "2025-01-09T12:02:00Z", "level": "ERROR", "logger": "atb.database", "message": "Database connection timeout", "component": "db_manager"}',
            '{"timestamp": "2025-01-09T12:03:00Z", "level": "INFO", "logger": "atb.strategy", "message": "Signal generated", "component": "ml_basic"}',
            '{"timestamp": "2025-01-09T12:04:00Z", "level": "CRITICAL", "logger": "atb.trading", "message": "Memory usage critical", "component": "trading_engine"}'
        ])
        
        # * Analyze logs
        analyzer = RailwayLogAnalyzer(db_manager=Mock())
        report = analyzer.analyze_logs(sample_logs, 24)
        
        # * Verify analysis results
        assert report.total_entries == 5
        assert report.error_count == 3  # ERROR + CRITICAL
        assert report.warning_count == 1
        assert report.critical_count == 1
        
        # * Should detect multiple error patterns
        assert len(report.error_patterns) >= 2
        
        # * Should have recommendations
        assert len(report.recommendations) > 0
        
        # * Generate markdown report
        markdown = analyzer.generate_report_markdown(report)
        assert "# Railway Log Analysis Report" in markdown
        assert "API rate limit" in markdown or "Database connection" in markdown

    @patch('subprocess.run')
    def test_railway_log_fetcher_mock(self, mock_subprocess):
        """Test Railway log fetcher with mocked subprocess calls."""
        # * Mock successful Railway CLI response
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "2025-01-09 12:00:00 INFO: Test log entry\n"
        mock_subprocess.return_value.stderr = ""
        
        # * Mock environment variables
        with patch.dict('os.environ', {
            'RAILWAY_PROJECT_ID': 'test-project-id',
            'RAILWAY_SERVICE_ID': 'test-service-id'
        }):
            fetcher = RailwayLogFetcher()
            
            # * Test log fetching
            log_content = fetcher.fetch_logs(hours=1, environment="staging")
            
            assert log_content == "2025-01-09 12:00:00 INFO: Test log entry\n"
            
            # * Verify Railway CLI was called correctly
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "railway" in call_args
            assert "logs" in call_args
            assert "--since" in call_args
            assert "1h" in call_args

    def test_error_pattern_detection_accuracy(self):
        """Test accuracy of error pattern detection with realistic log data."""
        # * Create realistic log scenarios
        rate_limit_logs = [
            '{"timestamp": "2025-01-09T12:00:00Z", "level": "ERROR", "message": "binance.exceptions.ClientError: APIError(code=-1003): Too many requests"}',
            '{"timestamp": "2025-01-09T12:01:00Z", "level": "ERROR", "message": "Rate limit exceeded, waiting 60 seconds"}',
            '{"timestamp": "2025-01-09T12:02:00Z", "level": "WARNING", "message": "HTTP 429 received from API endpoint"}'
        ]
        
        db_connection_logs = [
            '{"timestamp": "2025-01-09T12:05:00Z", "level": "CRITICAL", "message": "psycopg2.OperationalError: could not connect to server"}',
            '{"timestamp": "2025-01-09T12:06:00Z", "level": "ERROR", "message": "Database connection pool exhausted"}'
        ]
        
        json_parsing_logs = [
            '{"timestamp": "2025-01-09T12:10:00Z", "level": "ERROR", "message": "json.decoder.JSONDecodeError: Expecting value: line 1 column 1"}',
            '{"timestamp": "2025-01-09T12:11:00Z", "level": "ERROR", "message": "Invalid JSON response from API endpoint"}'
        ]
        
        all_logs = "\n".join(rate_limit_logs + db_connection_logs + json_parsing_logs)
        
        # * Analyze combined logs
        analyzer = RailwayLogAnalyzer(db_manager=Mock())
        report = analyzer.analyze_logs(all_logs, 24)
        
        # * Verify pattern detection
        pattern_ids = [p.pattern_id for p in report.error_patterns]
        
        assert "api_rate_limit" in pattern_ids
        assert "database_connection" in pattern_ids  
        assert "json_parsing" in pattern_ids
        
        # * Verify frequencies
        for pattern in report.error_patterns:
            if pattern.pattern_id == "api_rate_limit":
                assert pattern.frequency == 3
            elif pattern.pattern_id == "database_connection":
                assert pattern.frequency == 2
            elif pattern.pattern_id == "json_parsing":
                assert pattern.frequency == 2

    def test_report_persistence_and_retrieval(self):
        """Test saving and retrieving analysis reports."""
        sample_logs = '{"timestamp": "2025-01-09T12:00:00Z", "level": "ERROR", "message": "Test error for persistence"}'
        
        # * Create analyzer with mock database
        mock_db = Mock()
        mock_db.log_event.return_value = 456
        
        analyzer = RailwayLogAnalyzer(db_manager=mock_db)
        report = analyzer.analyze_logs(sample_logs, 24)
        
        # * Save to database
        event_id = analyzer.save_analysis_to_db(report)
        
        assert event_id == 456
        
        # * Verify database call
        mock_db.log_event.assert_called_once()
        call_kwargs = mock_db.log_event.call_args[1]
        
        assert call_kwargs["event_type"] == "ANALYSIS"
        assert "log analysis completed" in call_kwargs["message"].lower()
        assert call_kwargs["component"] == "log_analyzer"
        assert "details" in call_kwargs
        
        # * Verify details structure
        details = call_kwargs["details"]
        assert "total_entries" in details
        assert "error_count" in details
        assert "error_patterns" in details
        assert "recommendations" in details

    def test_markdown_report_format_compliance(self):
        """Test that generated markdown reports follow expected format."""
        sample_logs = "\n".join([
            '{"timestamp": "2025-01-09T12:00:00Z", "level": "ERROR", "message": "API rate limit exceeded"}',
            '{"timestamp": "2025-01-09T12:01:00Z", "level": "WARNING", "message": "Memory usage high"}',
            '{"timestamp": "2025-01-09T12:02:00Z", "level": "INFO", "message": "Normal operation"}'
        ])
        
        analyzer = RailwayLogAnalyzer(db_manager=Mock())
        report = analyzer.analyze_logs(sample_logs, 24)
        markdown = analyzer.generate_report_markdown(report)
        
        # * Verify required sections
        required_sections = [
            "# Railway Log Analysis Report",
            "## Summary", 
            "## Statistics",
            "**Total Log Entries:**",
            "**Errors:**",
            "**Warnings:**"
        ]
        
        for section in required_sections:
            assert section in markdown, f"Missing required section: {section}"
        
        # * Verify data accuracy in markdown
        assert "**Total Log Entries:** 3" in markdown
        assert "**Errors:** 1" in markdown
        assert "**Warnings:** 1" in markdown
        
        # * Should end with attribution
        assert "generated automatically" in markdown.lower()

    @patch.dict('os.environ', {
        'RAILWAY_PROJECT_ID': 'test-project',
        'RAILWAY_SERVICE_ID': 'test-service'
    })
    def test_fetcher_initialization(self):
        """Test Railway log fetcher initialization with environment variables."""
        fetcher = RailwayLogFetcher()
        
        assert fetcher.project_id == "test-project"
        assert fetcher.service_id == "test-service"

    def test_fetcher_initialization_without_env_vars(self):
        """Test that fetcher raises error without required environment variables."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="RAILWAY_PROJECT_ID must be set"):
                RailwayLogFetcher()

    def test_log_file_saving(self):
        """Test saving logs to timestamped files."""
        sample_content = "Test log content\nSecond line\nThird line"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            fetcher = RailwayLogFetcher()
            log_file = fetcher.save_logs_to_file(sample_content, temp_dir)
            
            # * Verify file was created
            assert log_file.exists()
            assert log_file.parent == Path(temp_dir)
            assert "railway_logs_" in log_file.name
            assert log_file.suffix == ".log"
            
            # * Verify content
            with open(log_file, 'r') as f:
                saved_content = f.read()
            assert saved_content == sample_content