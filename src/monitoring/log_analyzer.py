#!/usr/bin/env python3
"""
Railway Log Analyzer for Background Agent

Analyzes Railway logs to detect errors, warnings, and performance issues.
Generates concise reports and identifies actionable fixes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from src.database.manager import DatabaseManager
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Represents a parsed log entry from Railway."""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    component: str | None = None
    exception: str | None = None
    extra_fields: dict[str, Any] | None = None


@dataclass
class ErrorPattern:
    """Represents a detected error pattern in logs."""
    pattern_id: str
    error_type: str
    message_pattern: str
    severity: str
    frequency: int
    first_seen: datetime
    last_seen: datetime
    affected_components: list[str]
    sample_messages: list[str]
    suggested_fix: str | None = None


@dataclass
class LogAnalysisReport:
    """Complete analysis report of Railway logs."""
    analysis_date: datetime
    log_period_start: datetime
    log_period_end: datetime
    total_entries: int
    error_count: int
    warning_count: int
    critical_count: int
    error_patterns: list[ErrorPattern]
    performance_issues: list[str]
    recommendations: list[str]
    summary: str


class RailwayLogAnalyzer:
    """
    Analyzes Railway logs to identify errors, patterns, and performance issues.
    
    Features:
    - Parses structured JSON logs from Railway
    - Identifies error patterns and frequencies
    - Detects performance issues and anomalies
    - Generates actionable recommendations
    - Integrates with database for historical tracking
    """

    def __init__(self, db_manager: DatabaseManager | None = None):
        self.db_manager = db_manager or DatabaseManager()
        
        # ! Common error patterns that we can automatically detect and fix
        self.known_error_patterns = {
            "api_rate_limit": {
                "pattern": r"(?i)rate.?limit|429|too many requests",
                "severity": "warning",
                "fix_suggestion": "Implement exponential backoff and request throttling",
                "auto_fixable": True,
            },
            "database_connection": {
                "pattern": r"(?i)database.*(connection|timeout|refused)|psycopg2.*(connection|connect|operational)",
                "severity": "critical",
                "fix_suggestion": "Check database connection pool settings and health",
                "auto_fixable": False,
            },
            "api_authentication": {
                "pattern": r"(?i)authentication.*(failed|invalid)|unauthorized|401",
                "severity": "critical", 
                "fix_suggestion": "Verify API credentials and permissions",
                "auto_fixable": False,
            },
            "memory_issues": {
                "pattern": r"(?i)memory|out of memory|killed|oom",
                "severity": "critical",
                "fix_suggestion": "Increase memory allocation or optimize memory usage",
                "auto_fixable": False,
            },
            "timeout_errors": {
                "pattern": r"(?i)timeout|timed out|connection reset",
                "severity": "warning",
                "fix_suggestion": "Increase timeout values and implement retry logic",
                "auto_fixable": True,
            },
            "json_parsing": {
                "pattern": r"(?i)json.*decode|invalid json|json.*parse",
                "severity": "error",
                "fix_suggestion": "Add JSON validation and error handling",
                "auto_fixable": True,
            },
            "missing_config": {
                "pattern": r"(?i)config.*not found|environment variable.*not set|missing.*config",
                "severity": "error",
                "fix_suggestion": "Verify required environment variables are set",
                "auto_fixable": False,
            },
        }

    def parse_log_line(self, line: str) -> LogEntry | None:
        """
        Parse a single log line from Railway.
        
        Railway logs can be in JSON format (structured) or plain text.
        """
        line = line.strip()
        if not line:
            return None
            
        try:
            # * Try parsing as JSON first (structured logs)
            if line.startswith("{"):
                log_data = json.loads(line)
                return LogEntry(
                    timestamp=datetime.fromisoformat(log_data.get("timestamp", "").replace("Z", "+00:00")),
                    level=log_data.get("level", "INFO"),
                    logger_name=log_data.get("logger", "unknown"),
                    message=log_data.get("message", ""),
                    component=log_data.get("component"),
                    exception=log_data.get("exception"),
                    extra_fields={k: v for k, v in log_data.items() 
                                if k not in ["timestamp", "level", "logger", "message", "component", "exception"]}
                )
            else:
                # * Parse plain text logs with common patterns
                # * Pattern: YYYY-MM-DD HH:MM:SS LEVEL logger_name: message
                timestamp_pattern = r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"
                level_pattern = r"(DEBUG|INFO|WARNING|ERROR|CRITICAL)"
                
                match = re.match(
                    rf"^{timestamp_pattern}\s+{level_pattern}\s+([^:]+):\s*(.*)$",
                    line
                )
                
                if match:
                    timestamp_str, level, logger_name, message = match.groups()
                    
                    # * Parse timestamp with multiple format support
                    timestamp = self._parse_timestamp(timestamp_str)
                    
                    return LogEntry(
                        timestamp=timestamp,
                        level=level,
                        logger_name=logger_name,
                        message=message
                    )
                else:
                    # * Fallback: treat as unstructured message
                    return LogEntry(
                        timestamp=datetime.now(timezone.utc),
                        level="INFO",
                        logger_name="unstructured",
                        message=line
                    )
                    
        except Exception as e:
            logger.warning(f"Failed to parse log line: {e}")
            return None

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp with multiple format support."""
        # * Timezone-aware formats
        timezone_formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ", 
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
        ]
        
        # * Timezone-naive formats (assume UTC)
        naive_formats = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ]
        
        # * Try timezone-aware formats first
        for fmt in timezone_formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # * Try naive formats and add UTC timezone
        for fmt in naive_formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
                
        # * Fallback to current time if parsing fails
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return datetime.now(timezone.utc)

    def analyze_logs(self, log_content: str, period_hours: int = 24) -> LogAnalysisReport:
        """
        Analyze Railway logs and generate a comprehensive report.
        
        Args:
            log_content: Raw log content from Railway
            period_hours: Analysis period in hours (default: 24)
            
        Returns:
            LogAnalysisReport with findings and recommendations
        """
        configure_logging()
        logger.info(f"Starting log analysis for {period_hours}-hour period")
        
        # * Parse all log entries
        log_entries = []
        for line in log_content.split("\n"):
            entry = self.parse_log_line(line)
            if entry:
                log_entries.append(entry)
        
        if not log_entries:
            return self._create_empty_report()
            
        # * Filter entries within the analysis period
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=period_hours)
        recent_entries = [e for e in log_entries if e.timestamp >= cutoff_time]
        
        # * Categorize by severity
        error_entries = [e for e in recent_entries if e.level in ["ERROR", "CRITICAL"]]
        warning_entries = [e for e in recent_entries if e.level == "WARNING"]
        critical_entries = [e for e in recent_entries if e.level == "CRITICAL"]
        
        # * Detect error patterns
        error_patterns = self._detect_error_patterns(error_entries + warning_entries)
        
        # * Identify performance issues
        performance_issues = self._detect_performance_issues(recent_entries)
        
        # * Generate recommendations
        recommendations = self._generate_recommendations(error_patterns, performance_issues)
        
        # * Create summary
        summary = self._generate_summary(len(recent_entries), len(error_entries), 
                                       len(warning_entries), error_patterns)
        
        return LogAnalysisReport(
            analysis_date=datetime.now(),
            log_period_start=recent_entries[0].timestamp if recent_entries else cutoff_time,
            log_period_end=recent_entries[-1].timestamp if recent_entries else datetime.now(),
            total_entries=len(recent_entries),
            error_count=len(error_entries),
            warning_count=len(warning_entries),
            critical_count=len(critical_entries),
            error_patterns=error_patterns,
            performance_issues=performance_issues,
            recommendations=recommendations,
            summary=summary
        )

    def _detect_error_patterns(self, error_entries: list[LogEntry]) -> list[ErrorPattern]:
        """Detect recurring error patterns in log entries."""
        pattern_matches = {}
        
        for entry in error_entries:
            for pattern_id, pattern_config in self.known_error_patterns.items():
                if re.search(pattern_config["pattern"], entry.message, re.IGNORECASE):
                    if pattern_id not in pattern_matches:
                        pattern_matches[pattern_id] = {
                            "entries": [],
                            "components": set(),
                            "messages": set(),
                        }
                    
                    pattern_matches[pattern_id]["entries"].append(entry)
                    if entry.component:
                        pattern_matches[pattern_id]["components"].add(entry.component)
                    pattern_matches[pattern_id]["messages"].add(entry.message)
        
        # * Convert to ErrorPattern objects
        error_patterns = []
        for pattern_id, matches in pattern_matches.items():
            config = self.known_error_patterns[pattern_id]
            entries = matches["entries"]
            
            if entries:
                error_patterns.append(ErrorPattern(
                    pattern_id=pattern_id,
                    error_type=pattern_id.replace("_", " ").title(),
                    message_pattern=config["pattern"],
                    severity=config["severity"],
                    frequency=len(entries),
                    first_seen=min(e.timestamp for e in entries),
                    last_seen=max(e.timestamp for e in entries),
                    affected_components=list(matches["components"]),
                    sample_messages=list(matches["messages"])[:3],  # * Limit to 3 samples
                    suggested_fix=config["fix_suggestion"]
                ))
        
        return sorted(error_patterns, key=lambda x: x.frequency, reverse=True)

    def _detect_performance_issues(self, log_entries: list[LogEntry]) -> list[str]:
        """Detect performance-related issues in logs."""
        issues = []
        
        # * Check for high memory usage warnings
        memory_warnings = [e for e in log_entries if re.search(r"memory.*(high|usage|critical|consumption|threshold|alert|pressure)", e.message, re.IGNORECASE)]
        if len(memory_warnings) > 5:
            issues.append(f"High memory usage detected ({len(memory_warnings)} warnings)")
        
        # * Check for slow query warnings  
        slow_queries = [e for e in log_entries if re.search(r"slow.*query|query.*slow|execution.*time", e.message, re.IGNORECASE)]
        if len(slow_queries) > 3:
            issues.append(f"Slow database queries detected ({len(slow_queries)} instances)")
            
        # * Check for API latency issues
        latency_issues = [e for e in log_entries if re.search(r"latency|response.*time|slow.*api", e.message, re.IGNORECASE)]
        if len(latency_issues) > 10:
            issues.append(f"API latency issues detected ({len(latency_issues)} instances)")
            
        # * Check for restart/crash patterns
        restarts = [e for e in log_entries if re.search(r"restart|crash|killed|exit|shutdown", e.message, re.IGNORECASE)]
        if len(restarts) > 2:
            issues.append(f"Multiple service restarts detected ({len(restarts)} instances)")
        
        return issues

    def _generate_recommendations(self, error_patterns: list[ErrorPattern], 
                                performance_issues: list[str]) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # * Recommendations based on error patterns
        for pattern in error_patterns:
            if pattern.frequency >= 5:  # * High frequency errors
                recommendations.append(
                    f"🔥 HIGH PRIORITY: {pattern.error_type} occurred {pattern.frequency} times. "
                    f"Suggested fix: {pattern.suggested_fix}"
                )
            elif pattern.severity == "critical":
                recommendations.append(
                    f"🚨 CRITICAL: {pattern.error_type} needs immediate attention. "
                    f"Suggested fix: {pattern.suggested_fix}"
                )
        
        # * Recommendations based on performance issues
        for issue in performance_issues:
            recommendations.append(f"⚡ PERFORMANCE: {issue}")
        
        # * General recommendations
        if len(error_patterns) > 10:
            recommendations.append("📊 Consider implementing comprehensive error tracking and alerting")
            
        if not recommendations:
            recommendations.append("✅ No critical issues detected in the analyzed period")
            
        return recommendations

    def _generate_summary(self, total_entries: int, error_count: int, 
                         warning_count: int, error_patterns: list[ErrorPattern]) -> str:
        """Generate a concise summary of the log analysis."""
        if total_entries == 0:
            return "No log entries found in the analysis period."
            
        error_rate = (error_count / total_entries) * 100 if total_entries > 0 else 0
        
        summary_parts = [
            f"Analyzed {total_entries:,} log entries",
            f"Found {error_count} errors and {warning_count} warnings",
            f"Error rate: {error_rate:.2f}%"
        ]
        
        if error_patterns:
            top_pattern = error_patterns[0]
            summary_parts.append(
                f"Most frequent issue: {top_pattern.error_type} ({top_pattern.frequency} occurrences)"
            )
        
        return ". ".join(summary_parts) + "."

    def _create_empty_report(self) -> LogAnalysisReport:
        """Create an empty report when no logs are available."""
        return LogAnalysisReport(
            analysis_date=datetime.now(),
            log_period_start=datetime.now() - timedelta(hours=24),
            log_period_end=datetime.now(),
            total_entries=0,
            error_count=0,
            warning_count=0,
            critical_count=0,
            error_patterns=[],
            performance_issues=[],
            recommendations=["No logs available for analysis"],
            summary="No log entries found in the analysis period."
        )

    def save_analysis_to_db(self, report: LogAnalysisReport) -> int:
        """
        Save analysis report to database for historical tracking.
        
        Returns:
            Event ID of the saved analysis
        """
        try:
            details = {
                "total_entries": report.total_entries,
                "error_count": report.error_count,
                "warning_count": report.warning_count,
                "critical_count": report.critical_count,
                "error_patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "error_type": p.error_type,
                        "frequency": p.frequency,
                        "severity": p.severity,
                        "suggested_fix": p.suggested_fix,
                        "affected_components": p.affected_components,
                    }
                    for p in report.error_patterns
                ],
                "performance_issues": report.performance_issues,
                "recommendations": report.recommendations,
            }
            
            event_id = self.db_manager.log_event(
                event_type="ANALYSIS",
                message=f"Daily log analysis completed: {report.summary}",
                severity="info" if report.error_count == 0 else "warning",
                component="log_analyzer",
                details=details
            )
            
            logger.info(f"Saved log analysis report to database with event_id: {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to save analysis to database: {e}")
            return -1

    def generate_report_markdown(self, report: LogAnalysisReport) -> str:
        """Generate a markdown report for GitHub issues or PRs."""
        lines = [
            "# Railway Log Analysis Report",
            f"**Analysis Date:** {report.analysis_date.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Period:** {report.log_period_start.strftime('%Y-%m-%d %H:%M')} to {report.log_period_end.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Summary",
            report.summary,
            "",
            "## Statistics", 
            f"- **Total Log Entries:** {report.total_entries:,}",
            f"- **Errors:** {report.error_count}",
            f"- **Warnings:** {report.warning_count}",
            f"- **Critical Issues:** {report.critical_count}",
            "",
        ]
        
        if report.error_patterns:
            lines.extend([
                "## Error Patterns",
                "",
            ])
            
            for pattern in report.error_patterns:
                lines.extend([
                    f"### {pattern.error_type}",
                    f"- **Frequency:** {pattern.frequency} occurrences",
                    f"- **Severity:** {pattern.severity.upper()}",
                    f"- **First Seen:** {pattern.first_seen.strftime('%Y-%m-%d %H:%M')}",
                    f"- **Last Seen:** {pattern.last_seen.strftime('%Y-%m-%d %H:%M')}",
                ])
                
                if pattern.affected_components:
                    lines.append(f"- **Affected Components:** {', '.join(pattern.affected_components)}")
                    
                if pattern.suggested_fix:
                    lines.append(f"- **Suggested Fix:** {pattern.suggested_fix}")
                    
                if pattern.sample_messages:
                    lines.extend([
                        "- **Sample Messages:**",
                        *[f"  - `{msg[:100]}...`" if len(msg) > 100 else f"  - `{msg}`" 
                          for msg in pattern.sample_messages[:2]],
                    ])
                    
                lines.append("")
        
        if report.performance_issues:
            lines.extend([
                "## Performance Issues",
                "",
                *[f"- {issue}" for issue in report.performance_issues],
                "",
            ])
        
        if report.recommendations:
            lines.extend([
                "## Recommendations",
                "",
                *[f"- {rec}" for rec in report.recommendations],
                "",
            ])
        
        lines.extend([
            "---",
            "*This report was generated automatically by the Railway Log Analyzer*"
        ])
        
        return "\n".join(lines)


def analyze_railway_logs(log_content: str, period_hours: int = 24) -> LogAnalysisReport:
    """
    Convenience function to analyze Railway logs.
    
    Args:
        log_content: Raw log content from Railway
        period_hours: Analysis period in hours
        
    Returns:
        LogAnalysisReport with findings
    """
    analyzer = RailwayLogAnalyzer()
    return analyzer.analyze_logs(log_content, period_hours)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Railway logs")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    parser.add_argument("--period-hours", type=int, default=24, help="Analysis period in hours")
    parser.add_argument("--output", help="Output file for markdown report")
    
    args = parser.parse_args()
    
    # * Read log file
    try:
        with open(args.log_file) as f:
            log_content = f.read()
    except Exception as e:
        print(f"Error reading log file: {e}")
        import sys
        sys.exit(1)
    
    # * Analyze logs
    analyzer = RailwayLogAnalyzer()
    report = analyzer.analyze_logs(log_content, args.period_hours)
    
    # * Save to database
    analyzer.save_analysis_to_db(report)
    
    # * Generate markdown report
    markdown_report = analyzer.generate_report_markdown(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(markdown_report)
        print(f"Report saved to {args.output}")
    else:
        print(markdown_report)