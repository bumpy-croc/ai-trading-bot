"""
Rollback Validation and Testing Procedures

This module provides comprehensive validation and testing procedures for
rollback operations, including impact analysis, safety checks, and
emergency rollback procedures for production issues.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.strategies.base import BaseStrategy
from .rollback_manager import RollbackPoint


@dataclass
class RollbackValidationResult:
    """
    Result of rollback validation
    
    Attributes:
        validation_timestamp: When validation was performed
        rollback_id: ID of the rollback point being validated
        overall_status: Overall validation status (pass, fail, warning)
        safety_checks: Results of safety checks
        impact_analysis: Analysis of rollback impact
        recommendations: Recommendations based on validation
        risk_assessment: Risk assessment for the rollback
    """
    validation_timestamp: datetime
    rollback_id: str
    overall_status: str
    safety_checks: Dict[str, Any]
    impact_analysis: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'rollback_id': self.rollback_id,
            'overall_status': self.overall_status,
            'safety_checks': self.safety_checks,
            'impact_analysis': self.impact_analysis,
            'recommendations': self.recommendations,
            'risk_assessment': self.risk_assessment
        }


class RollbackValidator:
    """
    Comprehensive rollback validator
    
    This class provides validation and testing procedures for rollback operations,
    including safety checks, impact analysis, and emergency procedures.
    """
    
    def __init__(self):
        """Initialize the rollback validator"""
        self.logger = logging.getLogger("RollbackValidator")
        
        # Validation history
        self.validation_history: List[RollbackValidationResult] = []
    
    def validate_rollback_safety(self, rollback_point: RollbackPoint,
                                current_strategy: Optional[BaseStrategy] = None) -> RollbackValidationResult:
        """
        Perform comprehensive rollback safety validation
        
        Args:
            rollback_point: Rollback point to validate
            current_strategy: Current strategy instance for comparison
            
        Returns:
            RollbackValidationResult with detailed validation results
        """
        start_time = datetime.now()
        
        self.logger.info(f"Starting rollback safety validation for: {rollback_point.rollback_id}")
        
        # Perform safety checks
        safety_checks = self._perform_safety_checks(rollback_point)
        
        # Analyze rollback impact
        impact_analysis = self._analyze_rollback_impact(rollback_point, current_strategy)
        
        # Assess risk
        risk_assessment = self._assess_rollback_risk(rollback_point, safety_checks, impact_analysis)
        
        # Generate recommendations
        recommendations = self._generate_rollback_recommendations(
            safety_checks, impact_analysis, risk_assessment
        )
        
        # Determine overall status
        overall_status = self._determine_overall_status(safety_checks, impact_analysis, risk_assessment)
        
        # Create validation result
        result = RollbackValidationResult(
            validation_timestamp=start_time,
            rollback_id=rollback_point.rollback_id,
            overall_status=overall_status,
            safety_checks=safety_checks,
            impact_analysis=impact_analysis,
            recommendations=recommendations,
            risk_assessment=risk_assessment
        )
        
        # Store in history
        self.validation_history.append(result)
        
        self.logger.info(f"Rollback validation completed: {overall_status}")
        
        return result
    
    def _perform_safety_checks(self, rollback_point: RollbackPoint) -> Dict[str, Any]:
        """Perform comprehensive safety checks"""
        safety_checks = {
            'timestamp': datetime.now().isoformat(),
            'checks_performed': [],
            'passed_checks': [],
            'failed_checks': [],
            'warnings': [],
            'overall_safety': 'unknown'
        }
        
        # Check 1: Backup file integrity
        backup_check = self._check_backup_integrity(rollback_point)
        safety_checks['checks_performed'].append('backup_integrity')
        if backup_check['status'] == 'pass':
            safety_checks['passed_checks'].append('backup_integrity')
        else:
            safety_checks['failed_checks'].append('backup_integrity')
            safety_checks['warnings'].extend(backup_check.get('warnings', []))
        
        # Check 2: Rollback point age
        age_check = self._check_rollback_age(rollback_point)
        safety_checks['checks_performed'].append('rollback_age')
        if age_check['status'] == 'pass':
            safety_checks['passed_checks'].append('rollback_age')
        else:
            safety_checks['failed_checks'].append('rollback_age')
            safety_checks['warnings'].extend(age_check.get('warnings', []))
        
        # Determine overall safety
        if safety_checks['failed_checks']:
            safety_checks['overall_safety'] = 'unsafe'
        elif safety_checks['warnings']:
            safety_checks['overall_safety'] = 'caution'
        else:
            safety_checks['overall_safety'] = 'safe'
        
        return safety_checks
    
    def _analyze_rollback_impact(self, rollback_point: RollbackPoint,
                               current_strategy: Optional[BaseStrategy]) -> Dict[str, Any]:
        """Analyze the impact of performing the rollback"""
        impact_analysis = {
            'timestamp': datetime.now().isoformat(),
            'files_affected': len(rollback_point.file_backups),
            'configuration_changes': [],
            'data_loss_risk': 'low',
            'downtime_estimate': 0
        }
        
        # Analyze configuration changes
        if current_strategy:
            try:
                current_config = current_strategy.get_parameters() if hasattr(current_strategy, 'get_parameters') else {}
                legacy_config = rollback_point.legacy_strategy_config.get('parameters', {})
                
                config_changes = []
                for key, value in current_config.items():
                    if key in legacy_config and legacy_config[key] != value:
                        config_changes.append(f"{key}: {value} -> {legacy_config[key]}")
                
                impact_analysis['configuration_changes'] = config_changes
            except Exception as e:
                impact_analysis['configuration_changes'] = [f"Error analyzing config: {e}"]
        
        # Estimate downtime (simplified)
        base_downtime = 5  # 5 minutes base
        file_downtime = len(rollback_point.file_backups) * 1  # 1 minute per file
        impact_analysis['downtime_estimate'] = base_downtime + file_downtime
        
        # Assess data loss risk
        if rollback_point.converted_strategy_config:
            impact_analysis['data_loss_risk'] = 'medium'  # Converting back loses new features
        else:
            impact_analysis['data_loss_risk'] = 'low'  # Just reverting files
        
        return impact_analysis
    
    def _assess_rollback_risk(self, rollback_point: RollbackPoint,
                            safety_checks: Dict[str, Any],
                            impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk of performing rollback"""
        risk_assessment = {
            'timestamp': datetime.now().isoformat(),
            'overall_risk_level': 'unknown',
            'risk_factors': [],
            'go_no_go_recommendation': 'unknown'
        }
        
        risk_score = 0
        
        # Safety check risks
        if safety_checks['overall_safety'] == 'unsafe':
            risk_score += 3
            risk_assessment['risk_factors'].append('Safety checks failed')
        elif safety_checks['overall_safety'] == 'caution':
            risk_score += 1
            risk_assessment['risk_factors'].append('Safety checks have warnings')
        
        # Impact risks
        if impact_analysis['data_loss_risk'] == 'high':
            risk_score += 3
            risk_assessment['risk_factors'].append('High data loss risk')
        elif impact_analysis['data_loss_risk'] == 'medium':
            risk_score += 1
            risk_assessment['risk_factors'].append('Medium data loss risk')
        
        # Downtime risks
        if impact_analysis['downtime_estimate'] > 30:
            risk_score += 2
            risk_assessment['risk_factors'].append('Extended downtime expected')
        elif impact_analysis['downtime_estimate'] > 10:
            risk_score += 1
            risk_assessment['risk_factors'].append('Moderate downtime expected')
        
        # Determine risk level
        if risk_score >= 5:
            risk_assessment['overall_risk_level'] = 'high'
            risk_assessment['go_no_go_recommendation'] = 'no_go'
        elif risk_score >= 3:
            risk_assessment['overall_risk_level'] = 'medium'
            risk_assessment['go_no_go_recommendation'] = 'caution'
        elif risk_score >= 1:
            risk_assessment['overall_risk_level'] = 'low'
            risk_assessment['go_no_go_recommendation'] = 'go_with_caution'
        else:
            risk_assessment['overall_risk_level'] = 'minimal'
            risk_assessment['go_no_go_recommendation'] = 'go'
        
        return risk_assessment
    
    def _generate_rollback_recommendations(self, safety_checks: Dict[str, Any],
                                         impact_analysis: Dict[str, Any],
                                         risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Safety-based recommendations
        if safety_checks['overall_safety'] == 'unsafe':
            recommendations.append("DO NOT PROCEED: Safety checks failed. Address issues before rollback.")
        elif safety_checks['overall_safety'] == 'caution':
            recommendations.append("PROCEED WITH CAUTION: Safety checks have warnings.")
        
        # Risk-based recommendations
        if risk_assessment['go_no_go_recommendation'] == 'no_go':
            recommendations.append("RECOMMENDATION: Do not proceed with rollback due to high risk.")
        elif risk_assessment['go_no_go_recommendation'] == 'caution':
            recommendations.append("RECOMMENDATION: Proceed only if necessary, with careful monitoring.")
        
        # Impact-based recommendations
        if impact_analysis['downtime_estimate'] > 15:
            recommendations.append("Schedule rollback during maintenance window due to expected downtime.")
        
        if impact_analysis['data_loss_risk'] != 'low':
            recommendations.append("Create additional backups before proceeding with rollback.")
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append("Rollback appears safe to proceed with standard monitoring.")
        
        return recommendations
    
    def _determine_overall_status(self, safety_checks: Dict[str, Any],
                                impact_analysis: Dict[str, Any],
                                risk_assessment: Dict[str, Any]) -> str:
        """Determine overall validation status"""
        
        # Critical failures
        if (safety_checks['overall_safety'] == 'unsafe' or
            risk_assessment['go_no_go_recommendation'] == 'no_go'):
            return 'fail'
        
        # Warnings but acceptable
        if (safety_checks['overall_safety'] == 'caution' or
            risk_assessment['go_no_go_recommendation'] in ['caution', 'go_with_caution']):
            return 'warning'
        
        # All good
        return 'pass'
    
    def _check_backup_integrity(self, rollback_point: RollbackPoint) -> Dict[str, Any]:
        """Check integrity of backup files"""
        from pathlib import Path
        
        check_result = {
            'status': 'pass',
            'warnings': [],
            'missing_files': [],
            'corrupted_files': []
        }
        
        for original_path, backup_path in rollback_point.file_backups.items():
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                check_result['missing_files'].append(backup_path)
                check_result['status'] = 'fail'
            else:
                # Basic integrity check (file size > 0)
                try:
                    if backup_file.stat().st_size == 0:
                        check_result['corrupted_files'].append(backup_path)
                        check_result['status'] = 'fail'
                except Exception as e:
                    check_result['warnings'].append(f"Could not check {backup_path}: {e}")
        
        if check_result['missing_files']:
            check_result['warnings'].append(f"Missing backup files: {check_result['missing_files']}")
        
        if check_result['corrupted_files']:
            check_result['warnings'].append(f"Corrupted backup files: {check_result['corrupted_files']}")
        
        return check_result
    
    def _check_rollback_age(self, rollback_point: RollbackPoint) -> Dict[str, Any]:
        """Check if rollback point is too old"""
        age = datetime.now() - rollback_point.timestamp
        
        if age.days > 90:
            return {
                'status': 'fail',
                'warnings': [f"Rollback point is {age.days} days old (>90 days)"],
                'age_days': age.days
            }
        elif age.days > 30:
            return {
                'status': 'pass',
                'warnings': [f"Rollback point is {age.days} days old (>30 days)"],
                'age_days': age.days
            }
        else:
            return {
                'status': 'pass',
                'warnings': [],
                'age_days': age.days
            }
    
    def get_validation_history(self) -> List[RollbackValidationResult]:
        """Get validation history"""
        return self.validation_history.copy()
    
    def clear_validation_history(self) -> None:
        """Clear validation history"""
        self.validation_history.clear()
        self.logger.info("Rollback validation history cleared")