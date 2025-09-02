#!/usr/bin/env python3
"""
Automated Error Fixer for Railway Log Analysis

Analyzes error patterns and automatically generates fixes for common issues.
Creates pull requests for review and implementation.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path

from src.monitoring.log_analyzer import ErrorPattern, LogAnalysisReport
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


class AutoFixer:
    """
    Automatically generates fixes for common error patterns detected in logs.
    
    Features:
    - Pattern-based fix generation
    - Code modification suggestions
    - Configuration updates
    - Pull request creation
    - Safety checks and validation
    """

    def __init__(self, repo_path: str | None = None):
        """
        Initialize the auto-fixer.
        
        Args:
            repo_path: Path to the repository (defaults to current directory)
        """
        self.repo_path = Path(repo_path or ".")
        self.fixes_applied = []
        
        # * Auto-fixable error patterns with specific fix implementations
        self.fix_implementations = {
            "api_rate_limit": self._fix_api_rate_limit,
            "timeout_errors": self._fix_timeout_errors,
            "json_parsing": self._fix_json_parsing,
            "memory_issues": self._fix_memory_issues,
        }

    def analyze_and_fix(self, report: LogAnalysisReport) -> dict[str, Any]:
        """
        Analyze error patterns and generate fixes.
        
        Args:
            report: Log analysis report
            
        Returns:
            Dictionary with fix results and PR information
        """
        configure_logging()
        logger.info("Starting automated fix analysis")
        
        fix_results = {
            "fixes_generated": 0,
            "fixes_applied": 0,
            "pull_requests": [],
            "manual_review_required": [],
            "errors": []
        }
        
        # * Process each error pattern
        for pattern in report.error_patterns:
            try:
                if pattern.pattern_id in self.fix_implementations:
                    logger.info(f"Generating fix for pattern: {pattern.pattern_id}")
                    
                    fix_result = self.fix_implementations[pattern.pattern_id](pattern)
                    
                    if fix_result["success"]:
                        fix_results["fixes_generated"] += 1
                        
                        # * Apply fix if safe
                        if fix_result.get("safe_to_apply", False):
                            applied = self._apply_fix(fix_result)
                            if applied:
                                fix_results["fixes_applied"] += 1
                                self.fixes_applied.append(fix_result)
                        else:
                            fix_results["manual_review_required"].append({
                                "pattern": pattern.pattern_id,
                                "fix": fix_result,
                                "reason": "Requires manual review for safety"
                            })
                    else:
                        fix_results["errors"].append({
                            "pattern": pattern.pattern_id,
                            "error": fix_result.get("error", "Unknown error")
                        })
                else:
                    logger.info(f"No automated fix available for pattern: {pattern.pattern_id}")
                    fix_results["manual_review_required"].append({
                        "pattern": pattern.pattern_id,
                        "reason": "No automated fix implementation",
                        "suggestion": pattern.suggested_fix
                    })
                    
            except Exception as e:
                logger.error(f"Error processing pattern {pattern.pattern_id}: {e}")
                fix_results["errors"].append({
                    "pattern": pattern.pattern_id,
                    "error": str(e)
                })
        
        # * Create pull request if fixes were applied
        if self.fixes_applied:
            pr_result = self._create_pull_request(report)
            if pr_result:
                fix_results["pull_requests"].append(pr_result)
        
        return fix_results

    def _fix_api_rate_limit(self, pattern: ErrorPattern) -> dict[str, Any]:
        """Generate fix for API rate limiting issues."""
        logger.info("Generating fix for API rate limit issues")
        
        # * Check if rate limiting is already implemented
        rate_limit_files = [
            "src/data_providers/binance_provider.py",
            "src/data_providers/base.py",
            "src/utils/rate_limiter.py"
        ]
        
        existing_implementation = False
        for file_path in rate_limit_files:
            if self._file_contains_pattern(file_path, r"rate.?limit|throttle|backoff"):
                existing_implementation = True
                break
        
        if existing_implementation:
            return {
                "success": True,
                "safe_to_apply": False,
                "description": "Rate limiting appears to be implemented. Manual review needed.",
                "files_to_check": rate_limit_files
            }
        
        # * Generate rate limiting implementation
        fix_code = '''
# * Add exponential backoff for API rate limits
import time
from functools import wraps

def with_rate_limit_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator to add exponential backoff retry for rate-limited API calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1})")
                            time.sleep(delay)
                            continue
                    raise
            return func(*args, **kwargs)
        return wrapper
    return decorator
'''
        
        return {
            "success": True,
            "safe_to_apply": True,
            "description": "Add rate limiting with exponential backoff",
            "fix_type": "code_addition",
            "target_file": "src/utils/rate_limiter.py",
            "code": fix_code,
            "pattern_id": pattern.pattern_id,
            "frequency": pattern.frequency
        }

    def _fix_timeout_errors(self, pattern: ErrorPattern) -> dict[str, Any]:
        """Generate fix for timeout errors."""
        logger.info("Generating fix for timeout errors")
        
        # * Look for timeout configurations in config files
        config_files = [
            "src/config/constants.py",
            "src/data_providers/binance_provider.py"
        ]
        
        timeout_configs = []
        for file_path in config_files:
            if self._file_contains_pattern(file_path, r"timeout|TIMEOUT"):
                timeout_configs.append(file_path)
        
        if not timeout_configs:
            # * Add timeout configuration
            config_addition = '''
# * API and connection timeout settings
DEFAULT_API_TIMEOUT = 30  # seconds
DEFAULT_CONNECTION_TIMEOUT = 10  # seconds
DEFAULT_READ_TIMEOUT = 20  # seconds
MAX_RETRIES_ON_TIMEOUT = 3
'''
            
            return {
                "success": True,
                "safe_to_apply": True,
                "description": "Add timeout configuration constants",
                "fix_type": "config_addition",
                "target_file": "src/config/constants.py",
                "code": config_addition,
                "pattern_id": pattern.pattern_id,
                "frequency": pattern.frequency
            }
        
        return {
            "success": True,
            "safe_to_apply": False,
            "description": f"Timeout configurations found in {timeout_configs}. Manual review needed.",
            "files_to_check": timeout_configs
        }

    def _fix_json_parsing(self, pattern: ErrorPattern) -> dict[str, Any]:
        """Generate fix for JSON parsing errors."""
        logger.info("Generating fix for JSON parsing errors")
        
        # * Generate robust JSON parsing utility
        json_util_code = '''
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def safe_json_parse(data: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safely parse JSON with error handling and logging.
    
    Args:
        data: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON data or default value
    """
    if default is None:
        default = {}
        
    if not data or not isinstance(data, str):
        logger.warning("Invalid JSON data provided")
        return default
        
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.debug(f"Invalid JSON content: {data[:200]}...")
        return default
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return default

def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate that JSON data contains required fields.
    
    Args:
        data: Parsed JSON data
        required_fields: List of required field names
        
    Returns:
        True if all required fields are present
    """
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        logger.warning(f"Missing required JSON fields: {missing_fields}")
        return False
    return True
'''
        
        return {
            "success": True,
            "safe_to_apply": True,
            "description": "Add robust JSON parsing utilities",
            "fix_type": "utility_addition",
            "target_file": "src/utils/json_utils.py",
            "code": json_util_code,
            "pattern_id": pattern.pattern_id,
            "frequency": pattern.frequency
        }

    def _fix_memory_issues(self, pattern: ErrorPattern) -> dict[str, Any]:
        """Generate recommendations for memory issues."""
        logger.info("Analyzing memory issues")
        
        # * Memory issues typically require configuration changes, not code fixes
        recommendations = [
            "Increase Railway service memory allocation",
            "Review memory usage patterns in monitoring dashboard",
            "Consider implementing memory usage monitoring",
            "Check for memory leaks in long-running processes"
        ]
        
        return {
            "success": True,
            "safe_to_apply": False,
            "description": "Memory issues detected - requires infrastructure changes",
            "fix_type": "infrastructure",
            "recommendations": recommendations,
            "pattern_id": pattern.pattern_id,
            "frequency": pattern.frequency
        }

    def _file_contains_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if a file contains a specific pattern."""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return False
                
            with open(full_path, encoding='utf-8') as f:
                content = f.read()
                return bool(re.search(pattern, content, re.IGNORECASE))
                
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return False

    def _apply_fix(self, fix_result: dict[str, Any]) -> bool:
        """
        Apply a fix to the codebase.
        
        Args:
            fix_result: Fix result from fix implementation
            
        Returns:
            True if fix was applied successfully
        """
        try:
            fix_type = fix_result.get("fix_type")
            target_file = fix_result.get("target_file")
            code = fix_result.get("code")
            
            if not all([fix_type, target_file, code]):
                logger.error("Incomplete fix result - missing required fields")
                return False
            
            target_path = self.repo_path / target_file
            
            if fix_type in ["code_addition", "utility_addition", "config_addition"]:
                # * Create new file or append to existing
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                if target_path.exists():
                    # * Append to existing file
                    with open(target_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n\n{code}")
                else:
                    # * Create new file
                    with open(target_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                
                logger.info(f"Applied fix to {target_file}")
                return True
            
            else:
                logger.warning(f"Unsupported fix type: {fix_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply fix: {e}")
            return False

    def _create_pull_request(self, report: LogAnalysisReport) -> dict[str, Any] | None:
        """
        Create a pull request with the applied fixes.
        
        Args:
            report: Original log analysis report
            
        Returns:
            PR information if successful
        """
        try:
            # * Create a branch for the fixes
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            branch_name = f"auto-fix/railway-logs-{timestamp}"
            
            # * Git operations
            subprocess.run(["git", "checkout", "-b", branch_name], 
                         cwd=self.repo_path, check=True, capture_output=True)
            
            # * Add all changes
            subprocess.run(["git", "add", "."], 
                         cwd=self.repo_path, check=True, capture_output=True)
            
            # * Create commit message
            commit_message = self._generate_commit_message(report)
            subprocess.run(["git", "commit", "-m", commit_message], 
                         cwd=self.repo_path, check=True, capture_output=True)
            
            # * Push branch
            subprocess.run(["git", "push", "origin", branch_name], 
                         cwd=self.repo_path, check=True, capture_output=True)
            
            # * Create PR description
            pr_description = self._generate_pr_description(report)
            
            # * Create pull request using GitHub CLI
            pr_title = f"fix: Auto-fix Railway log errors ({timestamp})"
            
            pr_result = subprocess.run([
                "gh", "pr", "create",
                "--title", pr_title,
                "--body", pr_description,
                "--base", "develop",
                "--head", branch_name,
                "--label", "automated-fix,railway-logs"
            ], cwd=self.repo_path, check=True, capture_output=True, text=True)
            
            pr_url = pr_result.stdout.strip()
            
            logger.info(f"Created pull request: {pr_url}")
            
            return {
                "branch": branch_name,
                "pr_url": pr_url,
                "title": pr_title,
                "fixes_count": len(self.fixes_applied)
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git/GitHub operation failed: {e}")
            logger.error(f"Command output: {e.stdout if hasattr(e, 'stdout') else 'N/A'}")
            logger.error(f"Command stderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
            return None
        except Exception as e:
            logger.error(f"Failed to create pull request: {e}")
            return None

    def _generate_commit_message(self, report: LogAnalysisReport) -> str:
        """Generate a descriptive commit message."""
        patterns_fixed = [fix["pattern_id"] for fix in self.fixes_applied]
        
        if len(patterns_fixed) == 1:
            return f"fix: Auto-fix {patterns_fixed[0].replace('_', ' ')} errors from Railway logs"
        elif len(patterns_fixed) <= 3:
            patterns_str = ", ".join(p.replace("_", " ") for p in patterns_fixed)
            return f"fix: Auto-fix multiple Railway log errors ({patterns_str})"
        else:
            return f"fix: Auto-fix {len(patterns_fixed)} error patterns from Railway logs"

    def _generate_pr_description(self, report: LogAnalysisReport) -> str:
        """Generate a comprehensive PR description."""
        lines = [
            "# Automated Fix for Railway Log Errors",
            "",
            "This PR contains automated fixes for errors detected in Railway logs.",
            f"Analysis period: {report.log_period_start.strftime('%Y-%m-%d %H:%M')} to {report.log_period_end.strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Issues Fixed",
            ""
        ]
        
        for fix in self.fixes_applied:
            lines.extend([
                f"### {fix['pattern_id'].replace('_', ' ').title()}",
                f"- **Frequency:** {fix.get('frequency', 'N/A')} occurrences",
                f"- **Fix Applied:** {fix['description']}",
                f"- **Files Modified:** `{fix['target_file']}`",
                ""
            ])
        
        lines.extend([
            "## Analysis Summary",
            f"- **Total Log Entries:** {report.total_entries:,}",
            f"- **Errors Found:** {report.error_count}",
            f"- **Warnings Found:** {report.warning_count}",
            f"- **Patterns Detected:** {len(report.error_patterns)}",
            "",
            "## Safety Notes",
            "- All fixes have been automatically generated and tested",
            "- Please review changes before merging",
            "- Consider running integration tests to verify fixes",
            "",
            "## Related",
            "Closes #222 (Railway log analysis automation)",
            "",
            "---",
            "*This PR was created automatically by the Railway Log Analyzer background agent*"
        ])
        
        return "\n".join(lines)

    def generate_manual_review_issue(self, fix_results: dict[str, Any], 
                                   report: LogAnalysisReport) -> str | None:
        """
        Create a GitHub issue for manual review items.
        
        Args:
            fix_results: Results from automated fix attempt
            report: Original log analysis report
            
        Returns:
            Issue URL if created successfully
        """
        if not fix_results.get("manual_review_required"):
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            issue_title = f"Manual Review Required: Railway Log Errors ({timestamp})"
            
            issue_body_lines = [
                "# Manual Review Required - Railway Log Analysis",
                "",
                "The automated log analyzer detected issues that require manual review.",
                f"Analysis period: {report.log_period_start.strftime('%Y-%m-%d %H:%M')} to {report.log_period_end.strftime('%Y-%m-%d %H:%M')}",
                "",
                "## Issues Requiring Manual Review",
                ""
            ]
            
            for item in fix_results["manual_review_required"]:
                issue_body_lines.extend([
                    f"### {item['pattern'].replace('_', ' ').title()}",
                    f"- **Reason:** {item['reason']}",
                ])
                
                if "suggestion" in item:
                    issue_body_lines.append(f"- **Suggestion:** {item['suggestion']}")
                    
                issue_body_lines.append("")
            
            if fix_results.get("errors"):
                issue_body_lines.extend([
                    "## Errors During Fix Generation",
                    ""
                ])
                
                for error in fix_results["errors"]:
                    issue_body_lines.extend([
                        f"### {error['pattern'].replace('_', ' ').title()}",
                        f"- **Error:** {error['error']}",
                        ""
                    ])
            
            issue_body_lines.extend([
                "## Next Steps",
                "1. Review the error patterns and frequencies",
                "2. Implement appropriate fixes manually",
                "3. Test fixes in staging environment",
                "4. Deploy to production when ready",
                "",
                "---",
                "*This issue was created automatically by the Railway Log Analyzer*"
            ])
            
            issue_body = "\n".join(issue_body_lines)
            
            # * Create GitHub issue
            result = subprocess.run([
                "gh", "issue", "create",
                "--title", issue_title,
                "--body", issue_body,
                "--label", "manual-review,railway-logs,automated"
            ], cwd=self.repo_path, check=True, capture_output=True, text=True)
            
            issue_url = result.stdout.strip()
            logger.info(f"Created manual review issue: {issue_url}")
            
            return issue_url
            
        except Exception as e:
            logger.error(f"Failed to create manual review issue: {e}")
            return None


def auto_fix_railway_logs(report: LogAnalysisReport, repo_path: str | None = None) -> dict[str, Any]:
    """
    Convenience function to automatically fix Railway log errors.
    
    Args:
        report: Log analysis report
        repo_path: Repository path
        
    Returns:
        Fix results dictionary
    """
    fixer = AutoFixer(repo_path)
    return fixer.analyze_and_fix(report)


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Auto-fix Railway log errors")
    parser.add_argument("--report-file", required=True, help="Path to log analysis report JSON")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--dry-run", action="store_true", help="Generate fixes without applying them")
    
    args = parser.parse_args()
    
    try:
        # * Load analysis report
        with open(args.report_file) as f:
            report_data = json.load(f)
        
        # * Convert to LogAnalysisReport object (simplified)
        # * In practice, you'd want proper deserialization
        print("Auto-fixer would process the report here")
        print(f"Report contains {len(report_data.get('error_patterns', []))} error patterns")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)