#!/usr/bin/env python3
"""
Railway Log Fetcher

Fetches logs from Railway services using the Railway CLI and formats them
for analysis by the background agent.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config.config_manager import get_config
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


class RailwayLogFetcher:
    """
    Fetches logs from Railway services for analysis.
    
    Uses Railway CLI to retrieve logs and formats them for processing
    by the log analyzer and background agent.
    """

    def __init__(self, project_id: str | None = None, service_id: str | None = None):
        """
        Initialize Railway log fetcher.
        
        Args:
            project_id: Railway project ID (defaults to env var)
            service_id: Railway service ID (defaults to env var)
        """
        self.config = get_config()
        self.project_id = project_id or self.config.get("RAILWAY_PROJECT_ID")
        self.service_id = service_id or self.config.get("RAILWAY_SERVICE_ID") 
        
        if not self.project_id:
            raise ValueError("RAILWAY_PROJECT_ID must be set in environment or provided")

    def _run_railway_command(self, command: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
        """
        Run a Railway CLI command with error handling.
        
        Args:
            command: Railway CLI command as list
            timeout: Command timeout in seconds
            
        Returns:
            CompletedProcess result
            
        Raises:
            RuntimeError: If command fails or Railway CLI is not available
        """
        try:
            # * Set Railway project context
            env = {**os.environ, "RAILWAY_PROJECT_ID": self.project_id}
            if self.service_id:
                env["RAILWAY_SERVICE_ID"] = self.service_id
                
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False
            )
            
            if result.returncode != 0:
                error_msg = f"Railway command failed: {' '.join(command)}\nStderr: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            return result
            
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Railway command timed out after {timeout}s: {' '.join(command)}") from e
        except FileNotFoundError as e:
            raise RuntimeError("Railway CLI not found. Install with: curl -fsSL https://railway.app/install.sh | sh") from e

    def fetch_logs(self, hours: int = 24, environment: str = "production") -> str:
        """
        Fetch logs from Railway for the specified time period.
        
        Args:
            hours: Number of hours of logs to fetch (default: 24)
            environment: Railway environment name (default: production)
            
        Returns:
            Raw log content as string
        """
        configure_logging()
        logger.info(f"Fetching {hours} hours of logs from Railway environment: {environment}")
        
        try:
            # * Build Railway logs command
            command = ["railway", "logs"]
            
            # * Add time filter
            if hours <= 24:
                command.extend(["--since", f"{hours}h"])
            else:
                # * For longer periods, use days
                days = max(1, hours // 24)
                command.extend(["--since", f"{days}d"])
            
            # * Add environment filter if specified
            if environment and environment != "production":
                command.extend(["--environment", environment])
                
            # * Add service filter if available
            if self.service_id:
                command.extend(["--service", self.service_id])
            
            # * Execute command
            result = self._run_railway_command(command)
            
            if not result.stdout.strip():
                logger.warning("No logs retrieved from Railway")
                return ""
                
            logger.info(f"Successfully fetched {len(result.stdout.splitlines())} log lines")
            return result.stdout
            
        except Exception as e:
            logger.error(f"Failed to fetch Railway logs: {e}")
            raise

    def fetch_logs_with_filters(self, hours: int = 24, 
                              environment: str = "production",
                              filters: list[str] | None = None) -> dict[str, str]:
        """
        Fetch logs with multiple filters for targeted analysis.
        
        Args:
            hours: Number of hours of logs to fetch
            environment: Railway environment name
            filters: List of log filters (e.g., ["ERROR", "WARNING"])
            
        Returns:
            Dictionary mapping filter name to log content
        """
        if not filters:
            filters = ["ERROR", "WARNING", "CRITICAL"]
            
        log_results = {}
        
        for filter_term in filters:
            try:
                logger.info(f"Fetching logs with filter: {filter_term}")
                
                command = ["railway", "logs"]
                
                # * Add time filter
                if hours <= 24:
                    command.extend(["--since", f"{hours}h"])
                else:
                    days = max(1, hours // 24)
                    command.extend(["--since", f"{days}d"])
                
                # * Add environment and service filters
                if environment and environment != "production":
                    command.extend(["--environment", environment])
                if self.service_id:
                    command.extend(["--service", self.service_id])
                    
                # * Add content filter
                command.extend(["--filter", filter_term])
                
                result = self._run_railway_command(command)
                log_results[filter_term] = result.stdout
                
                logger.info(f"Fetched {len(result.stdout.splitlines())} lines for filter: {filter_term}")
                
            except Exception as e:
                logger.error(f"Failed to fetch logs with filter {filter_term}: {e}")
                log_results[filter_term] = ""
        
        return log_results

    def save_logs_to_file(self, log_content: str, output_dir: str = "logs/railway") -> Path:
        """
        Save fetched logs to a timestamped file.
        
        Args:
            log_content: Raw log content
            output_dir: Directory to save logs
            
        Returns:
            Path to saved log file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_path / f"railway_logs_{timestamp}.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)
            
        logger.info(f"Saved logs to: {log_file}")
        return log_file

    def get_service_info(self) -> dict[str, Any]:
        """
        Get Railway service information for context.
        
        Returns:
            Dictionary with service details
        """
        try:
            # * Get service status
            result = self._run_railway_command(["railway", "status", "--json"])
            service_info = json.loads(result.stdout)
            
            return {
                "project_id": self.project_id,
                "service_id": self.service_id,
                "environment": service_info.get("environment"),
                "deployment_id": service_info.get("deployment", {}).get("id"),
                "status": service_info.get("deployment", {}).get("status"),
                "last_deployment": service_info.get("deployment", {}).get("createdAt"),
            }
            
        except Exception as e:
            logger.warning(f"Could not fetch service info: {e}")
            return {
                "project_id": self.project_id,
                "service_id": self.service_id,
                "error": str(e)
            }


def fetch_railway_logs(hours: int = 24, environment: str = "production", 
                      save_to_file: bool = True) -> tuple[str, Path | None]:
    """
    Convenience function to fetch Railway logs.
    
    Args:
        hours: Number of hours of logs to fetch
        environment: Railway environment name
        save_to_file: Whether to save logs to file
        
    Returns:
        Tuple of (log_content, log_file_path)
    """
    fetcher = RailwayLogFetcher()
    log_content = fetcher.fetch_logs(hours, environment)
    
    log_file_path = None
    if save_to_file and log_content:
        log_file_path = fetcher.save_logs_to_file(log_content)
    
    return log_content, log_file_path


if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Fetch Railway logs")
    parser.add_argument("--hours", type=int, default=24, help="Hours of logs to fetch")
    parser.add_argument("--environment", default="production", help="Railway environment")
    parser.add_argument("--output-dir", default="logs/railway", help="Output directory")
    parser.add_argument("--filters", nargs="*", help="Log filters (ERROR, WARNING, etc.)")
    
    args = parser.parse_args()
    
    try:
        fetcher = RailwayLogFetcher()
        
        if args.filters:
            # * Fetch with specific filters
            log_results = fetcher.fetch_logs_with_filters(
                hours=args.hours,
                environment=args.environment, 
                filters=args.filters
            )
            
            for filter_name, content in log_results.items():
                if content:
                    output_file = fetcher.save_logs_to_file(
                        content, 
                        f"{args.output_dir}/{filter_name.lower()}"
                    )
                    print(f"Saved {filter_name} logs to: {output_file}")
        else:
            # * Fetch all logs
            log_content = fetcher.fetch_logs(args.hours, args.environment)
            if log_content:
                output_file = fetcher.save_logs_to_file(log_content, args.output_dir)
                print(f"Saved logs to: {output_file}")
            else:
                print("No logs retrieved")
                
    except Exception as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)