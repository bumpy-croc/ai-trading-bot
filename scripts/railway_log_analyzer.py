#!/usr/bin/env python3
"""
Daily Railway Log Analysis Script

Main orchestrator script that:
1. Fetches logs from Railway
2. Analyzes them for errors and patterns
3. Generates fixes for common issues
4. Creates pull requests for review

This script is designed to be run daily by a background agent or scheduler.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# * Add project root to path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(1, str(src_path))

from src.database.manager import DatabaseManager
from src.monitoring.auto_fixer import AutoFixer
from src.monitoring.log_analyzer import RailwayLogAnalyzer
from src.monitoring.railway_log_fetcher import RailwayLogFetcher
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


class DailyLogAnalysisOrchestrator:
    """
    Orchestrates the daily Railway log analysis process.
    
    Coordinates log fetching, analysis, fix generation, and PR creation
    to provide a complete automated error detection and fixing pipeline.
    """

    def __init__(self, 
                 repo_path: str = ".",
                 environment: str = "production",
                 analysis_hours: int = 24,
                 dry_run: bool = False):
        """
        Initialize the orchestrator.
        
        Args:
            repo_path: Repository root path
            environment: Railway environment to analyze
            analysis_hours: Hours of logs to analyze
            dry_run: If True, don't apply fixes or create PRs
        """
        self.repo_path = Path(repo_path)
        self.environment = environment
        self.analysis_hours = analysis_hours
        self.dry_run = dry_run
        
        # * Initialize components
        self.log_fetcher = RailwayLogFetcher()
        self.log_analyzer = RailwayLogAnalyzer()
        self.auto_fixer = AutoFixer(repo_path)
        self.db_manager = DatabaseManager()

    def run_daily_analysis(self) -> dict[str, Any]:
        """
        Run the complete daily log analysis pipeline.
        
        Returns:
            Results dictionary with analysis and fix information
        """
        configure_logging()
        logger.info("=" * 60)
        logger.info("STARTING DAILY RAILWAY LOG ANALYSIS")
        logger.info("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "analysis_hours": self.analysis_hours,
            "dry_run": self.dry_run,
            "steps": {},
            "errors": []
        }
        
        try:
            # * Step 1: Fetch Railway logs
            logger.info(f"Step 1: Fetching {self.analysis_hours}h of logs from {self.environment}")
            log_content, log_file_path = self._fetch_logs()
            results["steps"]["log_fetch"] = {
                "success": bool(log_content),
                "log_file": str(log_file_path) if log_file_path else None,
                "log_lines": len(log_content.splitlines()) if log_content else 0
            }
            
            if not log_content:
                logger.warning("No logs retrieved - ending analysis")
                return results
            
            # * Step 2: Analyze logs for errors and patterns
            logger.info("Step 2: Analyzing logs for error patterns")
            analysis_report = self.log_analyzer.analyze_logs(log_content, self.analysis_hours)
            results["steps"]["analysis"] = {
                "success": True,
                "total_entries": analysis_report.total_entries,
                "error_count": analysis_report.error_count,
                "warning_count": analysis_report.warning_count,
                "patterns_found": len(analysis_report.error_patterns)
            }
            
            # * Save analysis to database
            event_id = self.log_analyzer.save_analysis_to_db(analysis_report)
            results["steps"]["database_save"] = {
                "success": event_id > 0,
                "event_id": event_id
            }
            
            # * Step 3: Generate and apply fixes
            logger.info("Step 3: Generating automated fixes")
            if not self.dry_run and analysis_report.error_patterns:
                fix_results = self.auto_fixer.analyze_and_fix(analysis_report)
                results["steps"]["auto_fix"] = fix_results
                
                # * Create manual review issue if needed
                if fix_results.get("manual_review_required"):
                    issue_url = self.auto_fixer.generate_manual_review_issue(fix_results, analysis_report)
                    results["steps"]["manual_review_issue"] = {
                        "created": bool(issue_url),
                        "url": issue_url
                    }
            else:
                logger.info("Skipping fix generation (dry run or no patterns found)")
                results["steps"]["auto_fix"] = {"skipped": True, "reason": "dry_run or no_patterns"}
            
            # * Step 4: Generate reports
            logger.info("Step 4: Generating reports")
            markdown_report = self.log_analyzer.generate_report_markdown(analysis_report)
            
            # * Save markdown report
            report_file = self._save_report(markdown_report, analysis_report)
            results["steps"]["report_generation"] = {
                "success": True,
                "report_file": str(report_file)
            }
            
            # * Generate overview for logging
            overview = self._generate_log_overview(analysis_report)
            logger.info("=" * 60)
            logger.info("LOG ANALYSIS OVERVIEW")
            logger.info("=" * 60)
            logger.info(overview)
            logger.info("=" * 60)
            
            results["overview"] = overview
            results["success"] = True
            
        except Exception as e:
            logger.error(f"Daily analysis failed: {e}")
            results["errors"].append(str(e))
            results["success"] = False
        
        return results

    def _fetch_logs(self) -> tuple[str, Path | None]:
        """Fetch logs from Railway."""
        try:
            log_content = self.log_fetcher.fetch_logs(
                hours=self.analysis_hours,
                environment=self.environment
            )
            
            # * Save logs to file for reference
            log_file_path = None
            if log_content:
                log_file_path = self.log_fetcher.save_logs_to_file(
                    log_content, 
                    f"logs/railway/{self.environment}"
                )
            
            return log_content, log_file_path
            
        except Exception as e:
            logger.error(f"Failed to fetch Railway logs: {e}")
            return "", None

    def _save_report(self, markdown_report: str, analysis_report) -> Path:
        """Save the markdown report to file."""
        reports_dir = Path("logs/analysis_reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"railway_log_analysis_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        logger.info(f"Saved analysis report to: {report_file}")
        return report_file

    def _generate_log_overview(self, report) -> str:
        """Generate a concise log overview for console output."""
        if report.total_entries == 0:
            return "No log entries found in the analysis period."
        
        lines = [
            f"Period: {report.log_period_start.strftime('%Y-%m-%d %H:%M')} to {report.log_period_end.strftime('%Y-%m-%d %H:%M')}",
            f"Total Entries: {report.total_entries:,}",
            f"Errors: {report.error_count} | Warnings: {report.warning_count} | Critical: {report.critical_count}",
        ]
        
        if report.error_patterns:
            lines.append("")
            lines.append("Top Error Patterns:")
            for i, pattern in enumerate(report.error_patterns[:3], 1):
                lines.append(f"  {i}. {pattern.error_type}: {pattern.frequency} occurrences ({pattern.severity})")
        
        if report.performance_issues:
            lines.append("")
            lines.append("Performance Issues:")
            for issue in report.performance_issues[:3]:
                lines.append(f"  - {issue}")
        
        if report.recommendations:
            lines.append("")
            lines.append("Key Recommendations:")
            for rec in report.recommendations[:3]:
                lines.append(f"  - {rec}")
        
        return "\n".join(lines)


def main():
    """Main entry point for the daily log analysis script."""
    parser = argparse.ArgumentParser(description="Daily Railway Log Analysis")
    parser.add_argument("--environment", default="production", 
                       help="Railway environment to analyze")
    parser.add_argument("--hours", type=int, default=24, 
                       help="Hours of logs to analyze")
    parser.add_argument("--dry-run", action="store_true",
                       help="Analyze only, don't apply fixes or create PRs")
    parser.add_argument("--repo-path", default=".",
                       help="Repository root path")
    parser.add_argument("--output-json", 
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    try:
        # * Initialize and run orchestrator
        orchestrator = DailyLogAnalysisOrchestrator(
            repo_path=args.repo_path,
            environment=args.environment,
            analysis_hours=args.hours,
            dry_run=args.dry_run
        )
        
        results = orchestrator.run_daily_analysis()
        
        # * Save results to JSON if requested
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {args.output_json}")
        
        # * Print summary
        if results.get("success"):
            print("\n✅ Daily log analysis completed successfully")
            if "overview" in results:
                print("\nOverview:")
                print(results["overview"])
        else:
            print("\n❌ Daily log analysis failed")
            for error in results.get("errors", []):
                print(f"Error: {error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()