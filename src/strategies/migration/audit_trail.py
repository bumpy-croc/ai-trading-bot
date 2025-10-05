"""
Audit Trail Utilities for Strategy Migration

This module provides comprehensive audit trail functionality for tracking
strategy conversions, validations, and migrations with detailed logging
and reporting capabilities.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .strategy_converter import ConversionReport
from .validation_utils import ValidationReport


@dataclass
class AuditEvent:
    """
    Single audit event in the migration process
    
    Attributes:
        event_id: Unique identifier for the event
        timestamp: When the event occurred
        event_type: Type of event (conversion, validation, migration, etc.)
        strategy_name: Name of the strategy involved
        user_id: ID of the user who initiated the event
        operation: Specific operation performed
        status: Status of the operation (success, failure, warning)
        details: Detailed information about the event
        metadata: Additional metadata
    """
    event_id: str
    timestamp: datetime
    event_type: str
    strategy_name: str
    user_id: str
    operation: str
    status: str
    details: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "strategy_name": self.strategy_name,
            "user_id": self.user_id,
            "operation": self.operation,
            "status": self.status,
            "details": self.details,
            "metadata": self.metadata
        }


@dataclass
class MigrationSession:
    """
    Complete migration session tracking
    
    Attributes:
        session_id: Unique identifier for the migration session
        start_time: When the session started
        end_time: When the session ended (None if ongoing)
        user_id: ID of the user who initiated the session
        session_type: Type of migration session
        strategies_processed: List of strategy names processed
        events: List of audit events in this session
        status: Overall session status
        summary: Session summary information
    """
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    user_id: str
    session_type: str
    strategies_processed: List[str]
    events: List[AuditEvent]
    status: str
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "user_id": self.user_id,
            "session_type": self.session_type,
            "strategies_processed": self.strategies_processed,
            "events": [event.to_dict() for event in self.events],
            "status": self.status,
            "summary": self.summary
        }


class AuditTrailManager:
    """
    Comprehensive audit trail manager for strategy migration
    
    This class provides functionality to track, log, and report on all
    migration activities with detailed audit trails for compliance
    and debugging purposes.
    """

    def __init__(self, audit_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the audit trail manager
        
        Args:
            audit_dir: Directory to store audit files (default: ./audit_trails)
        """
        self.logger = logging.getLogger("AuditTrailManager")

        # Set up audit directory
        self.audit_dir = Path(audit_dir) if audit_dir else Path("audit_trails")
        self.audit_dir.mkdir(exist_ok=True)

        # Initialize storage
        self.events: List[AuditEvent] = []
        self.sessions: List[MigrationSession] = []
        self.current_session: Optional[MigrationSession] = None

        # Event counter for unique IDs
        self._event_counter = 0

        # Load existing audit data
        self._load_audit_data()

        self.logger.info(f"AuditTrailManager initialized with audit directory: {self.audit_dir}")

    def start_migration_session(self, user_id: str, session_type: str = "manual") -> str:
        """
        Start a new migration session
        
        Args:
            user_id: ID of the user starting the session
            session_type: Type of session (manual, automated, batch, etc.)
            
        Returns:
            Session ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.sessions)}"

        session = MigrationSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            user_id=user_id,
            session_type=session_type,
            strategies_processed=[],
            events=[],
            status="active",
            summary={}
        )

        self.current_session = session
        self.sessions.append(session)

        # Log session start event
        self.log_event(
            event_type="session",
            strategy_name="",
            user_id=user_id,
            operation="session_start",
            status="success",
            details={
                "session_id": session_id,
                "session_type": session_type
            }
        )

        self.logger.info(f"Started migration session: {session_id}")
        return session_id

    def end_migration_session(self, session_id: Optional[str] = None) -> None:
        """
        End a migration session
        
        Args:
            session_id: ID of the session to end (uses current session if None)
        """
        session = self._get_session(session_id)
        if not session:
            self.logger.warning(f"Session not found: {session_id}")
            return

        session.end_time = datetime.now()
        session.status = "completed"

        # Generate session summary
        session.summary = self._generate_session_summary(session)

        # Log session end event
        self.log_event(
            event_type="session",
            strategy_name="",
            user_id=session.user_id,
            operation="session_end",
            status="success",
            details={
                "session_id": session.session_id,
                "duration_minutes": (session.end_time - session.start_time).total_seconds() / 60,
                "strategies_processed": len(session.strategies_processed),
                "events_logged": len(session.events)
            }
        )

        # Clear current session if it's the one being ended
        if self.current_session and self.current_session.session_id == session.session_id:
            self.current_session = None

        # Save audit data
        self._save_audit_data()

        self.logger.info(f"Ended migration session: {session.session_id}")

    def log_event(self, event_type: str, strategy_name: str, user_id: str,
                  operation: str, status: str, details: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an audit event
        
        Args:
            event_type: Type of event (conversion, validation, migration, etc.)
            strategy_name: Name of the strategy involved
            user_id: ID of the user who initiated the event
            operation: Specific operation performed
            status: Status of the operation (success, failure, warning)
            details: Detailed information about the event
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        self._event_counter += 1
        event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._event_counter}"

        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            strategy_name=strategy_name,
            user_id=user_id,
            operation=operation,
            status=status,
            details=details,
            metadata=metadata or {}
        )

        # Add to global events list
        self.events.append(event)

        # Add to current session if active
        if self.current_session:
            self.current_session.events.append(event)
            if strategy_name and strategy_name not in self.current_session.strategies_processed:
                self.current_session.strategies_processed.append(strategy_name)

        self.logger.debug(f"Logged audit event: {event_id} - {operation} for {strategy_name}")
        return event_id

    def log_conversion(self, conversion_report: ConversionReport, user_id: str) -> str:
        """
        Log a strategy conversion event
        
        Args:
            conversion_report: Conversion report from strategy converter
            user_id: ID of the user who initiated the conversion
            
        Returns:
            Event ID
        """
        return self.log_event(
            event_type="conversion",
            strategy_name=conversion_report.strategy_name,
            user_id=user_id,
            operation="strategy_conversion",
            status="success" if conversion_report.success else "failure",
            details={
                "source_strategy_type": conversion_report.source_strategy_type,
                "target_components": conversion_report.target_components,
                "parameter_mappings": conversion_report.parameter_mappings,
                "validation_results": conversion_report.validation_results,
                "warnings": conversion_report.warnings,
                "errors": conversion_report.errors,
                "audit_trail": conversion_report.audit_trail
            },
            metadata={
                "conversion_timestamp": conversion_report.conversion_timestamp.isoformat(),
                "conversion_duration": "unknown"  # Could be calculated if needed
            }
        )

    def log_validation(self, validation_report: ValidationReport, user_id: str) -> str:
        """
        Log a strategy validation event
        
        Args:
            validation_report: Validation report from strategy validator
            user_id: ID of the user who initiated the validation
            
        Returns:
            Event ID
        """
        return self.log_event(
            event_type="validation",
            strategy_name=validation_report.strategy_name,
            user_id=user_id,
            operation="strategy_validation",
            status="success" if validation_report.overall_success else "failure",
            details={
                "total_tests": validation_report.total_tests,
                "passed_tests": validation_report.passed_tests,
                "failed_tests": validation_report.failed_tests,
                "test_results": [result.to_dict() for result in validation_report.test_results],
                "recommendations": validation_report.recommendations,
                "performance_metrics": validation_report.performance_metrics
            },
            metadata={
                "validation_timestamp": validation_report.validation_timestamp.isoformat(),
                "success_rate": (validation_report.passed_tests / validation_report.total_tests) * 100 if validation_report.total_tests > 0 else 0
            }
        )

    def log_migration_step(self, strategy_name: str, user_id: str, step: str,
                          status: str, details: Dict[str, Any]) -> str:
        """
        Log a migration step
        
        Args:
            strategy_name: Name of the strategy being migrated
            user_id: ID of the user performing the migration
            step: Migration step name
            status: Status of the step
            details: Step details
            
        Returns:
            Event ID
        """
        return self.log_event(
            event_type="migration",
            strategy_name=strategy_name,
            user_id=user_id,
            operation=f"migration_step_{step}",
            status=status,
            details=details
        )

    def get_events_for_strategy(self, strategy_name: str) -> List[AuditEvent]:
        """
        Get all audit events for a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            List of audit events
        """
        return [event for event in self.events if event.strategy_name == strategy_name]

    def get_events_by_type(self, event_type: str) -> List[AuditEvent]:
        """
        Get all audit events of a specific type
        
        Args:
            event_type: Type of events to retrieve
            
        Returns:
            List of audit events
        """
        return [event for event in self.events if event.event_type == event_type]

    def get_events_by_user(self, user_id: str) -> List[AuditEvent]:
        """
        Get all audit events for a specific user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of audit events
        """
        return [event for event in self.events if event.user_id == user_id]

    def get_events_in_timerange(self, start_time: datetime, end_time: datetime) -> List[AuditEvent]:
        """
        Get all audit events within a time range
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            
        Returns:
            List of audit events
        """
        return [
            event for event in self.events
            if start_time <= event.timestamp <= end_time
        ]

    def generate_audit_report(self, strategy_name: Optional[str] = None,
                            user_id: Optional[str] = None,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive audit report
        
        Args:
            strategy_name: Filter by strategy name (optional)
            user_id: Filter by user ID (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            
        Returns:
            Audit report dictionary
        """
        # Filter events based on criteria
        filtered_events = self.events.copy()

        if strategy_name:
            filtered_events = [e for e in filtered_events if e.strategy_name == strategy_name]

        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]

        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]

        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]

        # Generate statistics
        total_events = len(filtered_events)
        event_types = {}
        operations = {}
        statuses = {}
        users = {}
        strategies = {}

        for event in filtered_events:
            # Count by event type
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

            # Count by operation
            operations[event.operation] = operations.get(event.operation, 0) + 1

            # Count by status
            statuses[event.status] = statuses.get(event.status, 0) + 1

            # Count by user
            users[event.user_id] = users.get(event.user_id, 0) + 1

            # Count by strategy
            if event.strategy_name:
                strategies[event.strategy_name] = strategies.get(event.strategy_name, 0) + 1

        # Calculate time range
        if filtered_events:
            earliest_event = min(filtered_events, key=lambda e: e.timestamp)
            latest_event = max(filtered_events, key=lambda e: e.timestamp)
            time_range = {
                "start": earliest_event.timestamp.isoformat(),
                "end": latest_event.timestamp.isoformat(),
                "duration_hours": (latest_event.timestamp - earliest_event.timestamp).total_seconds() / 3600
            }
        else:
            time_range = None

        return {
            "report_generated": datetime.now().isoformat(),
            "filters": {
                "strategy_name": strategy_name,
                "user_id": user_id,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None
            },
            "summary": {
                "total_events": total_events,
                "unique_strategies": len(strategies),
                "unique_users": len(users),
                "time_range": time_range
            },
            "statistics": {
                "event_types": event_types,
                "operations": operations,
                "statuses": statuses,
                "users": users,
                "strategies": strategies
            },
            "events": [event.to_dict() for event in filtered_events[-100:]]  # Last 100 events
        }

    def export_audit_trail(self, format: str = "json",
                          filename: Optional[str] = None) -> str:
        """
        Export audit trail to file
        
        Args:
            format: Export format (json, csv)
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_trail_{timestamp}.{format}"

        filepath = self.audit_dir / filename

        if format.lower() == "json":
            audit_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_events": len(self.events),
                "total_sessions": len(self.sessions),
                "events": [event.to_dict() for event in self.events],
                "sessions": [session.to_dict() for session in self.sessions]
            }

            with open(filepath, "w") as f:
                json.dump(audit_data, f, indent=2, default=str)

        elif format.lower() == "csv":
            # Export events as CSV
            events_data = []
            for event in self.events:
                events_data.append({
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "strategy_name": event.strategy_name,
                    "user_id": event.user_id,
                    "operation": event.operation,
                    "status": event.status,
                    "details": json.dumps(event.details),
                    "metadata": json.dumps(event.metadata)
                })

            df = pd.DataFrame(events_data)
            df.to_csv(filepath, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Exported audit trail to: {filepath}")
        return str(filepath)

    def import_audit_trail(self, filepath: Union[str, Path]) -> None:
        """
        Import audit trail from file
        
        Args:
            filepath: Path to the audit trail file
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Audit trail file not found: {filepath}")

        if filepath.suffix.lower() == ".json":
            with open(filepath) as f:
                audit_data = json.load(f)

            # Import events
            for event_data in audit_data.get("events", []):
                event = AuditEvent(
                    event_id=event_data["event_id"],
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    event_type=event_data["event_type"],
                    strategy_name=event_data["strategy_name"],
                    user_id=event_data["user_id"],
                    operation=event_data["operation"],
                    status=event_data["status"],
                    details=event_data["details"],
                    metadata=event_data["metadata"]
                )
                self.events.append(event)

            # Import sessions
            for session_data in audit_data.get("sessions", []):
                session = MigrationSession(
                    session_id=session_data["session_id"],
                    start_time=datetime.fromisoformat(session_data["start_time"]),
                    end_time=datetime.fromisoformat(session_data["end_time"]) if session_data["end_time"] else None,
                    user_id=session_data["user_id"],
                    session_type=session_data["session_type"],
                    strategies_processed=session_data["strategies_processed"],
                    events=[],  # Events are loaded separately
                    status=session_data["status"],
                    summary=session_data["summary"]
                )
                self.sessions.append(session)

        else:
            raise ValueError(f"Unsupported import format: {filepath.suffix}")

        self.logger.info(f"Imported audit trail from: {filepath}")

    def _get_session(self, session_id: Optional[str]) -> Optional[MigrationSession]:
        """Get session by ID or current session"""
        if session_id is None:
            return self.current_session

        for session in self.sessions:
            if session.session_id == session_id:
                return session

        return None

    def _generate_session_summary(self, session: MigrationSession) -> Dict[str, Any]:
        """Generate summary for a migration session"""
        events_by_type = {}
        events_by_status = {}

        for event in session.events:
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1
            events_by_status[event.status] = events_by_status.get(event.status, 0) + 1

        duration = (session.end_time - session.start_time).total_seconds() / 60 if session.end_time else 0

        return {
            "duration_minutes": duration,
            "strategies_processed": len(session.strategies_processed),
            "total_events": len(session.events),
            "events_by_type": events_by_type,
            "events_by_status": events_by_status,
            "success_rate": (events_by_status.get("success", 0) / len(session.events)) * 100 if session.events else 0
        }

    def _save_audit_data(self) -> None:
        """Save audit data to persistent storage"""
        try:
            audit_file = self.audit_dir / "audit_data.json"
            audit_data = {
                "last_updated": datetime.now().isoformat(),
                "events": [event.to_dict() for event in self.events],
                "sessions": [session.to_dict() for session in self.sessions]
            }

            with open(audit_file, "w") as f:
                json.dump(audit_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save audit data: {e}")

    def _load_audit_data(self) -> None:
        """Load audit data from persistent storage"""
        try:
            audit_file = self.audit_dir / "audit_data.json"

            if audit_file.exists():
                with open(audit_file) as f:
                    audit_data = json.load(f)

                # Load events
                for event_data in audit_data.get("events", []):
                    event = AuditEvent(
                        event_id=event_data["event_id"],
                        timestamp=datetime.fromisoformat(event_data["timestamp"]),
                        event_type=event_data["event_type"],
                        strategy_name=event_data["strategy_name"],
                        user_id=event_data["user_id"],
                        operation=event_data["operation"],
                        status=event_data["status"],
                        details=event_data["details"],
                        metadata=event_data["metadata"]
                    )
                    self.events.append(event)

                # Load sessions
                for session_data in audit_data.get("sessions", []):
                    session = MigrationSession(
                        session_id=session_data["session_id"],
                        start_time=datetime.fromisoformat(session_data["start_time"]),
                        end_time=datetime.fromisoformat(session_data["end_time"]) if session_data["end_time"] else None,
                        user_id=session_data["user_id"],
                        session_type=session_data["session_type"],
                        strategies_processed=session_data["strategies_processed"],
                        events=[],  # Events are loaded separately
                        status=session_data["status"],
                        summary=session_data["summary"]
                    )
                    self.sessions.append(session)

                # Update event counter
                if self.events:
                    self._event_counter = len(self.events)

                self.logger.info(f"Loaded {len(self.events)} events and {len(self.sessions)} sessions from audit data")

        except Exception as e:
            self.logger.warning(f"Failed to load audit data: {e}")

    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get comprehensive audit statistics"""
        if not self.events:
            return {
                "total_events": 0,
                "total_sessions": 0,
                "event_types": {},
                "operations": {},
                "statuses": {},
                "users": {},
                "strategies": {}
            }

        # Calculate statistics
        event_types = {}
        operations = {}
        statuses = {}
        users = {}
        strategies = {}

        for event in self.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            operations[event.operation] = operations.get(event.operation, 0) + 1
            statuses[event.status] = statuses.get(event.status, 0) + 1
            users[event.user_id] = users.get(event.user_id, 0) + 1
            if event.strategy_name:
                strategies[event.strategy_name] = strategies.get(event.strategy_name, 0) + 1

        return {
            "total_events": len(self.events),
            "total_sessions": len(self.sessions),
            "active_sessions": len([s for s in self.sessions if s.status == "active"]),
            "event_types": event_types,
            "operations": operations,
            "statuses": statuses,
            "users": users,
            "strategies": strategies,
            "success_rate": (statuses.get("success", 0) / len(self.events)) * 100 if self.events else 0
        }
