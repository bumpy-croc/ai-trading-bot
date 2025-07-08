"""
Performance monitoring utilities for the trading system.

Provides timing metrics, memory usage tracking, and performance analysis.
"""

import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    operation: str
    duration_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    cpu_percent: float
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """Performance monitoring and timing utilities"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process(os.getpid())
        
    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations"""
        if not self.enabled:
            yield
            return
            
        # Get initial state
        memory_before = self.get_memory_usage()
        cpu_before = self.process.cpu_percent()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # Calculate metrics
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            memory_after = self.get_memory_usage()
            memory_delta = memory_after - memory_before
            cpu_percent = self.process.cpu_percent()
            
            # Store metrics
            metric = PerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_delta_mb=memory_delta,
                cpu_percent=cpu_percent
            )
            
            self.metrics.append(metric)
            
            # Log if significant operation
            if duration_ms > 100:  # Log operations taking > 100ms
                logger.debug(f"Performance: {operation} took {duration_ms:.1f}ms, "
                           f"memory: {memory_delta:+.1f}MB, CPU: {cpu_percent:.1f}%")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of performance metrics"""
        if not self.metrics:
            return {}
        
        # Group by operation
        operations = {}
        for metric in self.metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric)
        
        # Calculate summaries
        summary = {}
        for op, metrics_list in operations.items():
            durations = [m.duration_ms for m in metrics_list]
            memory_deltas = [m.memory_delta_mb for m in metrics_list]
            
            summary[op] = {
                'count': len(metrics_list),
                'total_duration_ms': sum(durations),
                'avg_duration_ms': sum(durations) / len(durations),
                'max_duration_ms': max(durations),
                'min_duration_ms': min(durations),
                'total_memory_delta_mb': sum(memory_deltas),
                'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
            }
        
        return summary
    
    def log_summary(self):
        """Log performance summary"""
        if not self.enabled or not self.metrics:
            return
            
        summary = self.get_metrics_summary()
        
        logger.info("=== Performance Summary ===")
        for operation, stats in summary.items():
            logger.info(f"{operation}: {stats['count']} calls, "
                       f"avg: {stats['avg_duration_ms']:.1f}ms, "
                       f"total: {stats['total_duration_ms']:.1f}ms, "
                       f"memory: {stats['avg_memory_delta_mb']:+.1f}MB")
    
    def clear_metrics(self):
        """Clear stored metrics"""
        self.metrics.clear()
    
    def get_current_system_stats(self) -> Dict:
        """Get current system performance stats"""
        try:
            return {
                'memory_usage_mb': self.get_memory_usage(),
                'cpu_percent': self.process.cpu_percent(),
                'memory_percent': self.process.memory_percent(),
                'num_threads': self.process.num_threads(),
                'num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else None
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}


# Global performance monitor instance
_global_monitor = PerformanceMonitor(enabled=True)


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _global_monitor


def set_monitoring_enabled(enabled: bool):
    """Enable or disable performance monitoring globally"""
    global _global_monitor
    _global_monitor.enabled = enabled


def timer(operation: str):
    """Decorator for timing function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with _global_monitor.timer(operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def performance_context(operation: str):
    """Context manager for performance monitoring"""
    with _global_monitor.timer(operation):
        yield


def log_performance_summary():
    """Log performance summary"""
    _global_monitor.log_summary()


def clear_performance_metrics():
    """Clear performance metrics"""
    _global_monitor.clear_metrics()