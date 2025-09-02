"""
Unit tests for Railway log analyzer.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.monitoring.log_analyzer import LogEntry, RailwayLogAnalyzer


class TestRailwayLogAnalyzer:
    """Test cases for RailwayLogAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = RailwayLogAnalyzer(db_manager=Mock())

    def test_parse_json_log_line(self):
        """Test parsing structured JSON log lines."""
        json_log = json.dumps({
            "timestamp": "2025-01-09T12:00:00Z",
            "level": "ERROR", 
            "logger": "atb.trading_engine",
            "message": "API rate limit exceeded",
            "component": "data_provider"
        })
        
        entry = self.analyzer.parse_log_line(json_log)
        
        assert entry is not None
        assert entry.level == "ERROR"
        assert entry.logger_name == "atb.trading_engine"
        assert entry.message == "API rate limit exceeded"
        assert entry.component == "data_provider"

    def test_parse_plain_text_log_line(self):
        """Test parsing plain text log lines."""
        plain_log = "2025-01-09 12:00:00 ERROR atb.trading_engine: Connection timeout to exchange"
        
        entry = self.analyzer.parse_log_line(plain_log)
        
        assert entry is not None
        assert entry.level == "ERROR"
        assert entry.logger_name == "atb.trading_engine"
        assert entry.message == "Connection timeout to exchange"

    def test_detect_api_rate_limit_pattern(self):
        """Test detection of API rate limit errors."""
        log_content = "\n".join([
            '{"timestamp": "2025-01-09T12:00:00Z", "level": "ERROR", "message": "API rate limit exceeded"}',
            '{"timestamp": "2025-01-09T12:01:00Z", "level": "ERROR", "message": "429 Too Many Requests"}',
            '{"timestamp": "2025-01-09T12:02:00Z", "level": "WARNING", "message": "Rate limit approaching"}'
        ])
        
        report = self.analyzer.analyze_logs(log_content, 24)
        
        # * Should detect rate limit pattern
        rate_limit_patterns = [p for p in report.error_patterns if p.pattern_id == "api_rate_limit"]
        assert len(rate_limit_patterns) == 1
        
        pattern = rate_limit_patterns[0]
        assert pattern.frequency == 3  # All three messages match
        assert pattern.severity == "warning"

    def test_detect_database_connection_pattern(self):
        """Test detection of database connection errors."""
        log_content = "\n".join([
            '{"timestamp": "2025-01-09T12:00:00Z", "level": "CRITICAL", "message": "Database connection refused"}',
            '{"timestamp": "2025-01-09T12:01:00Z", "level": "ERROR", "message": "psycopg2 connection timeout"}'
        ])
        
        report = self.analyzer.analyze_logs(log_content, 24)
        
        # * Should detect database connection pattern
        db_patterns = [p for p in report.error_patterns if p.pattern_id == "database_connection"]
        assert len(db_patterns) == 1
        
        pattern = db_patterns[0]
        assert pattern.frequency == 2
        assert pattern.severity == "critical"

    def test_performance_issue_detection(self):
        """Test detection of performance issues."""
        log_content = "\n".join([
            '{"timestamp": "2025-01-09T12:00:00Z", "level": "WARNING", "message": "Memory usage high"}',
            '{"timestamp": "2025-01-09T12:01:00Z", "level": "WARNING", "message": "Memory usage critical"}',
            '{"timestamp": "2025-01-09T12:02:00Z", "level": "WARNING", "message": "High memory consumption detected"}',
            '{"timestamp": "2025-01-09T12:03:00Z", "level": "WARNING", "message": "Memory threshold exceeded"}',
            '{"timestamp": "2025-01-09T12:04:00Z", "level": "WARNING", "message": "Memory usage alert"}',
            '{"timestamp": "2025-01-09T12:05:00Z", "level": "WARNING", "message": "Memory pressure detected"}'
        ])
        
        report = self.analyzer.analyze_logs(log_content, 24)
        
        # * Should detect memory performance issue
        memory_issues = [issue for issue in report.performance_issues if "memory" in issue.lower()]
        assert len(memory_issues) > 0

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        log_content = "\n".join([
            '{"timestamp": "2025-01-09T12:00:00Z", "level": "ERROR", "message": "API rate limit exceeded"}',
            '{"timestamp": "2025-01-09T12:01:00Z", "level": "ERROR", "message": "Rate limit hit again"}',
            '{"timestamp": "2025-01-09T12:02:00Z", "level": "ERROR", "message": "Too many requests"}',
            '{"timestamp": "2025-01-09T12:03:00Z", "level": "ERROR", "message": "429 error received"}',
            '{"timestamp": "2025-01-09T12:04:00Z", "level": "ERROR", "message": "Rate limiting active"}'
        ])
        
        report = self.analyzer.analyze_logs(log_content, 24)
        
        # * Should generate high priority recommendation for frequent rate limiting
        high_priority_recs = [r for r in report.recommendations if "HIGH PRIORITY" in r]
        assert len(high_priority_recs) > 0
        assert "rate limit" in high_priority_recs[0].lower()

    def test_empty_logs_handling(self):
        """Test handling of empty or invalid log content."""
        report = self.analyzer.analyze_logs("", 24)
        
        assert report.total_entries == 0
        assert report.error_count == 0
        assert len(report.error_patterns) == 0
        assert "No log entries found" in report.summary

    def test_markdown_report_generation(self):
        """Test markdown report generation."""
        log_content = '{"timestamp": "2025-01-09T12:00:00Z", "level": "ERROR", "message": "Test error"}'
        
        report = self.analyzer.analyze_logs(log_content, 24)
        markdown = self.analyzer.generate_report_markdown(report)
        
        assert "# Railway Log Analysis Report" in markdown
        assert "## Summary" in markdown
        assert "## Statistics" in markdown
        assert report.summary in markdown

    @patch('src.monitoring.log_analyzer.DatabaseManager')
    def test_database_integration(self, mock_db_manager):
        """Test database integration for saving analysis results."""
        mock_db_instance = Mock()
        mock_db_instance.log_event.return_value = 123
        mock_db_manager.return_value = mock_db_instance
        
        analyzer = RailwayLogAnalyzer(mock_db_instance)
        
        log_content = '{"timestamp": "2025-01-09T12:00:00Z", "level": "ERROR", "message": "Test error"}'
        report = analyzer.analyze_logs(log_content, 24)
        
        event_id = analyzer.save_analysis_to_db(report)
        
        assert event_id == 123
        mock_db_instance.log_event.assert_called_once()
        
        # * Verify the call arguments
        call_args = mock_db_instance.log_event.call_args
        assert call_args[1]["event_type"] == "ANALYSIS"
        assert "Daily log analysis completed" in call_args[1]["message"]