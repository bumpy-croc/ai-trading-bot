"""
Log analysis commands for the AI Trading Bot CLI.

Provides commands to fetch, analyze, and process Railway logs.
"""

from __future__ import annotations

import argparse
import json
import sys

# * Ensure project root and src are in sys.path for absolute imports
from src.utils.project_paths import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))

from src.monitoring.log_analyzer import RailwayLogAnalyzer
from src.monitoring.railway_log_fetcher import RailwayLogFetcher


def _print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + title)
    print("=" * max(8, len(title)))


def _logs_fetch(ns: argparse.Namespace) -> int:
    """Handle logs fetch command."""
    _print_header(f"Fetching Railway Logs - {ns.environment.upper()}")
    
    try:
        fetcher = RailwayLogFetcher()
        
        if ns.filters:
            # * Fetch with filters
            log_results = fetcher.fetch_logs_with_filters(
                hours=ns.hours,
                environment=ns.environment,
                filters=ns.filters
            )
            
            total_lines = 0
            for filter_name, content in log_results.items():
                if content:
                    lines = len(content.splitlines())
                    total_lines += lines
                    print(f"📄 {filter_name}: {lines:,} lines")
                    
                    if ns.save:
                        log_file = fetcher.save_logs_to_file(
                            content, 
                            f"logs/railway/{ns.environment}/{filter_name.lower()}"
                        )
                        print(f"   Saved to: {log_file}")
            
            print(f"\n✅ Total: {total_lines:,} log lines fetched")
        else:
            # * Fetch all logs
            log_content = fetcher.fetch_logs(ns.hours, ns.environment)
            
            if log_content:
                lines = len(log_content.splitlines())
                print(f"📄 Fetched {lines:,} log lines")
                
                if ns.save:
                    log_file = fetcher.save_logs_to_file(log_content, f"logs/railway/{ns.environment}")
                    print(f"📁 Saved to: {log_file}")
                
                if not ns.save:
                    # * Display recent logs if not saving
                    print("\n--- Recent Log Entries ---")
                    recent_lines = log_content.splitlines()[-10:]
                    for line in recent_lines:
                        print(line)
            else:
                print("❌ No logs retrieved")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to fetch logs: {e}")
        return 1


def _logs_analyze(ns: argparse.Namespace) -> int:
    """Handle logs analyze command."""
    _print_header("Analyzing Railway Logs")
    
    try:
        # * Read log content
        if ns.log_file:
            # * Analyze from file
            with open(ns.log_file) as f:
                log_content = f.read()
            print(f"📄 Analyzing log file: {ns.log_file}")
        else:
            # * Fetch and analyze
            fetcher = RailwayLogFetcher()
            log_content = fetcher.fetch_logs(ns.hours, ns.environment)
            print(f"📄 Fetched and analyzing {ns.hours}h of {ns.environment} logs")
        
        if not log_content:
            print("❌ No log content to analyze")
            return 1
        
        # * Analyze logs
        analyzer = RailwayLogAnalyzer()
        report = analyzer.analyze_logs(log_content, ns.hours)
        
        # * Save to database
        event_id = analyzer.save_analysis_to_db(report)
        print(f"💾 Saved analysis to database (event_id: {event_id})")
        
        # * Generate and save report
        markdown_report = analyzer.generate_report_markdown(report)
        
        if ns.output:
            with open(ns.output, 'w') as f:
                f.write(markdown_report)
            print(f"📄 Report saved to: {ns.output}")
        
        # * Print summary
        print("\n--- Analysis Summary ---")
        print(f"Period: {report.log_period_start.strftime('%Y-%m-%d %H:%M')} to {report.log_period_end.strftime('%Y-%m-%d %H:%M')}")
        print(f"Total Entries: {report.total_entries:,}")
        print(f"Errors: {report.error_count} | Warnings: {report.warning_count}")
        print(f"Error Patterns: {len(report.error_patterns)}")
        
        if report.error_patterns:
            print("\n--- Top Error Patterns ---")
            for i, pattern in enumerate(report.error_patterns[:3], 1):
                print(f"{i}. {pattern.error_type}: {pattern.frequency} occurrences ({pattern.severity})")
        
        if report.recommendations:
            print("\n--- Key Recommendations ---")
            for rec in report.recommendations[:3]:
                print(f"• {rec}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


def _logs_fix(ns: argparse.Namespace) -> int:
    """Handle logs fix command."""
    _print_header("Automated Error Fixing")
    
    try:
        # * Load analysis report
        if not ns.report_file:
            print("❌ Report file required for fixing")
            return 1
            
        with open(ns.report_file) as f:
            report_data = json.load(f)
        
        # * For now, show what would be fixed
        # * In a full implementation, you'd deserialize the report properly
        print(f"📊 Loaded analysis report from: {ns.report_file}")
        
        error_patterns = report_data.get("error_patterns", [])
        print(f"🔍 Found {len(error_patterns)} error patterns")
        
        if ns.dry_run:
            print("\n🔍 DRY RUN - No changes will be made")
            
        for pattern in error_patterns:
            pattern_id = pattern.get("pattern_id", "unknown")
            frequency = pattern.get("frequency", 0)
            suggested_fix = pattern.get("suggested_fix", "No suggestion available")
            
            print(f"\n📝 Pattern: {pattern_id}")
            print(f"   Frequency: {frequency}")
            print(f"   Suggested Fix: {suggested_fix}")
            
            if not ns.dry_run:
                print("   Status: Would generate fix (implementation pending)")
        
        if not ns.dry_run:
            print("\n⚠️  Automated fixing is not yet fully implemented")
            print("   This command currently shows what would be fixed")
        
        return 0
        
    except Exception as e:
        print(f"❌ Fix generation failed: {e}")
        return 1


def _logs_daily(ns: argparse.Namespace) -> int:
    """Handle logs daily command (full pipeline)."""
    _print_header("Daily Railway Log Analysis Pipeline")
    
    try:
        from scripts.railway_log_analyzer import DailyLogAnalysisOrchestrator
        
        # * Initialize orchestrator
        orchestrator = DailyLogAnalysisOrchestrator(
            repo_path=ns.repo_path,
            environment=ns.environment,
            analysis_hours=ns.hours,
            dry_run=ns.dry_run
        )
        
        # * Run full pipeline
        results = orchestrator.run_daily_analysis()
        
        # * Print results
        if results.get("success"):
            print("✅ Daily analysis completed successfully")
            
            if "overview" in results:
                print("\n--- Overview ---")
                print(results["overview"])
                
            if "steps" in results:
                steps = results["steps"]
                print("\n--- Pipeline Results ---")
                print(f"Log Fetch: {'✅' if steps.get('log_fetch', {}).get('success') else '❌'}")
                print(f"Analysis: {'✅' if steps.get('analysis', {}).get('success') else '❌'}")
                print(f"Auto Fix: {'✅' if steps.get('auto_fix', {}).get('fixes_applied', 0) > 0 else '⏭️'}")
                
                if steps.get("auto_fix", {}).get("pull_requests"):
                    for pr in steps["auto_fix"]["pull_requests"]:
                        print(f"🔧 Created PR: {pr['pr_url']}")
        else:
            print("❌ Daily analysis failed")
            for error in results.get("errors", []):
                print(f"   Error: {error}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Daily analysis pipeline failed: {e}")
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register log analysis commands."""
    p = subparsers.add_parser("logs", help="Railway log analysis commands")
    sub = p.add_subparsers(dest="logs_cmd", required=True)

    # * Fetch command
    p_fetch = sub.add_parser("fetch", help="Fetch logs from Railway")
    p_fetch.add_argument("--environment", default="production", 
                        choices=["production", "staging", "development"],
                        help="Railway environment")
    p_fetch.add_argument("--hours", type=int, default=24, 
                        help="Hours of logs to fetch")
    p_fetch.add_argument("--filters", nargs="*", 
                        help="Log filters (ERROR, WARNING, etc.)")
    p_fetch.add_argument("--save", action="store_true",
                        help="Save logs to files")
    p_fetch.set_defaults(func=_logs_fetch)

    # * Analyze command  
    p_analyze = sub.add_parser("analyze", help="Analyze Railway logs for errors")
    p_analyze.add_argument("--log-file", help="Analyze specific log file")
    p_analyze.add_argument("--environment", default="production",
                          choices=["production", "staging", "development"], 
                          help="Railway environment (if fetching)")
    p_analyze.add_argument("--hours", type=int, default=24,
                          help="Hours of logs to analyze")
    p_analyze.add_argument("--output", help="Output file for markdown report")
    p_analyze.set_defaults(func=_logs_analyze)

    # * Fix command
    p_fix = sub.add_parser("fix", help="Generate automated fixes for log errors")
    p_fix.add_argument("--report-file", required=True,
                      help="JSON report file from analysis")
    p_fix.add_argument("--dry-run", action="store_true",
                      help="Show fixes without applying them")
    p_fix.set_defaults(func=_logs_fix)

    # * Daily command (full pipeline)
    p_daily = sub.add_parser("daily", help="Run complete daily log analysis pipeline")
    p_daily.add_argument("--environment", default="production",
                        choices=["production", "staging", "development"],
                        help="Railway environment")
    p_daily.add_argument("--hours", type=int, default=24,
                        help="Hours of logs to analyze")
    p_daily.add_argument("--dry-run", action="store_true",
                        help="Analyze only, don't apply fixes")
    p_daily.add_argument("--repo-path", default=".",
                        help="Repository root path")
    p_daily.set_defaults(func=_logs_daily)