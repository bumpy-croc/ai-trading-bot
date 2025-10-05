"""
Strategy Conversion Utilities

This module provides utilities to convert existing legacy strategies to component-based
strategies, including parameter mapping, configuration conversion, and validation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.components.position_sizer import (
    ConfidenceWeightedSizer,
    FixedFractionSizer,
    PositionSizer,
)
from src.strategies.components.regime_context import EnhancedRegimeDetector
from src.strategies.components.risk_manager import (
    FixedRiskManager,
    RegimeAdaptiveRiskManager,
    RiskManager,
    VolatilityRiskManager,
)
from src.strategies.components import (
    MLBasicSignalGenerator,
    MLSignalGenerator,
    SignalGenerator,
    TechnicalSignalGenerator,
)
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter


@dataclass
class ConversionReport:
    """
    Report of strategy conversion process
    
    Attributes:
        strategy_name: Name of the converted strategy
        conversion_timestamp: When the conversion was performed
        source_strategy_type: Type of the source legacy strategy
        target_components: Dictionary of target component types
        parameter_mappings: Dictionary of parameter mappings applied
        validation_results: Results of conversion validation
        warnings: List of warnings encountered during conversion
        errors: List of errors encountered during conversion
        success: Whether the conversion was successful
        audit_trail: Detailed audit trail of conversion steps
    """
    strategy_name: str
    conversion_timestamp: datetime
    source_strategy_type: str
    target_components: Dict[str, str]
    parameter_mappings: Dict[str, Any]
    validation_results: Dict[str, bool]
    warnings: List[str]
    errors: List[str]
    success: bool
    audit_trail: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        return {
            "strategy_name": self.strategy_name,
            "conversion_timestamp": self.conversion_timestamp.isoformat(),
            "source_strategy_type": self.source_strategy_type,
            "target_components": self.target_components,
            "parameter_mappings": self.parameter_mappings,
            "validation_results": self.validation_results,
            "warnings": self.warnings,
            "errors": self.errors,
            "success": self.success,
            "audit_trail": self.audit_trail
        }


@dataclass
class ComponentMapping:
    """
    Mapping configuration for converting strategy to components
    
    Attributes:
        signal_generator_type: Type of signal generator to use
        risk_manager_type: Type of risk manager to use
        position_sizer_type: Type of position sizer to use
        parameter_mappings: Dictionary mapping legacy parameters to component parameters
        component_configs: Configuration for each component
    """
    signal_generator_type: Type[SignalGenerator]
    risk_manager_type: Type[RiskManager]
    position_sizer_type: Type[PositionSizer]
    parameter_mappings: Dict[str, Dict[str, str]]
    component_configs: Dict[str, Dict[str, Any]]


class StrategyConverter:
    """
    Utility class for converting legacy strategies to component-based strategies
    
    This class provides automated conversion from existing strategies to the new
    component-based architecture with comprehensive validation and reporting.
    """

    def __init__(self):
        """Initialize the strategy converter"""
        self.logger = logging.getLogger("StrategyConverter")

        # Define conversion mappings for known strategy types
        self._conversion_mappings = self._initialize_conversion_mappings()

        # Track conversion history
        self.conversion_history: List[ConversionReport] = []

    def convert_strategy(self, legacy_strategy: BaseStrategy,
                        target_name: Optional[str] = None,
                        custom_mapping: Optional[ComponentMapping] = None,
                        validate_conversion: bool = True) -> tuple[LegacyStrategyAdapter, ConversionReport]:
        """
        Convert a legacy strategy to component-based strategy
        
        Args:
            legacy_strategy: The legacy strategy to convert
            target_name: Name for the converted strategy (auto-generated if None)
            custom_mapping: Custom component mapping (uses default if None)
            validate_conversion: Whether to validate the conversion
            
        Returns:
            Tuple of (converted strategy adapter, conversion report)
        """
        start_time = datetime.now()
        strategy_type = legacy_strategy.__class__.__name__

        # Initialize conversion report
        report = ConversionReport(
            strategy_name=target_name or f"converted_{strategy_type}",
            conversion_timestamp=start_time,
            source_strategy_type=strategy_type,
            target_components={},
            parameter_mappings={},
            validation_results={},
            warnings=[],
            errors=[],
            success=False,
            audit_trail=[]
        )

        try:
            self.logger.info(f"Starting conversion of {strategy_type} strategy")
            report.audit_trail.append(f"Started conversion of {strategy_type} at {start_time}")

            # Step 1: Determine component mapping
            mapping = custom_mapping or self._get_default_mapping(legacy_strategy)
            if mapping is None:
                error_msg = f"No conversion mapping available for strategy type: {strategy_type}"
                report.errors.append(error_msg)
                self.logger.error(error_msg)
                return None, report

            report.target_components = {
                "signal_generator": mapping.signal_generator_type.__name__,
                "risk_manager": mapping.risk_manager_type.__name__,
                "position_sizer": mapping.position_sizer_type.__name__
            }
            report.audit_trail.append(f"Selected component mapping: {report.target_components}")

            # Step 2: Extract parameters from legacy strategy
            legacy_params = self._extract_legacy_parameters(legacy_strategy)
            report.audit_trail.append(f"Extracted legacy parameters: {list(legacy_params.keys())}")

            # Step 3: Map parameters to components
            component_params = self._map_parameters(legacy_params, mapping)
            report.parameter_mappings = component_params
            report.audit_trail.append(f"Mapped parameters to components")

            # Step 4: Create components
            components = self._create_components(mapping, component_params, report)
            if not all(components.values()):
                error_msg = "Failed to create one or more components"
                report.errors.append(error_msg)
                self.logger.error(error_msg)
                return None, report

            report.audit_trail.append("Successfully created all components")

            # Step 5: Create adapter strategy
            adapter_name = target_name or f"converted_{legacy_strategy.name}"
            adapter = LegacyStrategyAdapter(
                signal_generator=components["signal_generator"],
                risk_manager=components["risk_manager"],
                position_sizer=components["position_sizer"],
                regime_detector=EnhancedRegimeDetector(),
                name=adapter_name
            )

            # Copy trading pair and other basic properties
            adapter.set_trading_pair(legacy_strategy.get_trading_pair())

            report.audit_trail.append(f"Created adapter strategy: {adapter_name}")

            # Step 6: Validate conversion if requested
            if validate_conversion:
                validation_results = self._validate_conversion(legacy_strategy, adapter, report)
                report.validation_results = validation_results

                if not all(validation_results.values()):
                    report.warnings.append("Some validation checks failed")
                    self.logger.warning("Conversion validation had failures")

            # Mark conversion as successful
            report.success = True
            report.audit_trail.append(f"Conversion completed successfully at {datetime.now()}")

            # Store in history
            self.conversion_history.append(report)

            self.logger.info(f"Successfully converted {strategy_type} to component-based strategy")

            return adapter, report

        except Exception as e:
            error_msg = f"Conversion failed with error: {str(e)}"
            report.errors.append(error_msg)
            report.audit_trail.append(error_msg)
            self.logger.error(error_msg, exc_info=True)

            return None, report

    def batch_convert_strategies(self, strategies: List[BaseStrategy],
                               validate_conversions: bool = True) -> List[tuple[Optional[LegacyStrategyAdapter], ConversionReport]]:
        """
        Convert multiple strategies in batch
        
        Args:
            strategies: List of legacy strategies to convert
            validate_conversions: Whether to validate each conversion
            
        Returns:
            List of (converted strategy, conversion report) tuples
        """
        results = []

        self.logger.info(f"Starting batch conversion of {len(strategies)} strategies")

        for i, strategy in enumerate(strategies):
            self.logger.info(f"Converting strategy {i+1}/{len(strategies)}: {strategy.name}")

            try:
                adapter, report = self.convert_strategy(
                    strategy,
                    validate_conversion=validate_conversions
                )
                results.append((adapter, report))

            except Exception as e:
                # Create error report for failed conversion
                error_report = ConversionReport(
                    strategy_name=strategy.name,
                    conversion_timestamp=datetime.now(),
                    source_strategy_type=strategy.__class__.__name__,
                    target_components={},
                    parameter_mappings={},
                    validation_results={},
                    warnings=[],
                    errors=[f"Batch conversion failed: {str(e)}"],
                    success=False,
                    audit_trail=[f"Batch conversion error: {str(e)}"]
                )
                results.append((None, error_report))
                self.logger.error(f"Failed to convert strategy {strategy.name}: {e}")

        successful_conversions = sum(1 for adapter, _ in results if adapter is not None)
        self.logger.info(f"Batch conversion completed: {successful_conversions}/{len(strategies)} successful")

        return results

    def get_supported_strategy_types(self) -> List[str]:
        """
        Get list of strategy types that can be automatically converted
        
        Returns:
            List of supported strategy class names
        """
        return list(self._conversion_mappings.keys())

    def get_conversion_history(self) -> List[ConversionReport]:
        """
        Get history of all conversions performed
        
        Returns:
            List of conversion reports
        """
        return self.conversion_history.copy()

    def clear_conversion_history(self) -> None:
        """Clear the conversion history"""
        self.conversion_history.clear()
        self.logger.info("Conversion history cleared")

    def _initialize_conversion_mappings(self) -> Dict[str, ComponentMapping]:
        """Initialize default conversion mappings for known strategy types"""
        mappings = {}

        # ML Basic Strategy mapping
        mappings["MlBasic"] = ComponentMapping(
            signal_generator_type=MLBasicSignalGenerator,
            risk_manager_type=FixedRiskManager,
            position_sizer_type=ConfidenceWeightedSizer,
            parameter_mappings={
                "signal_generator": {
                    "model_path": "model_path",
                    "sequence_length": "sequence_length",
                    "use_prediction_engine": "use_prediction_engine",
                    "model_name": "model_name"
                },
                "risk_manager": {
                    "stop_loss_pct": "stop_loss_percentage",
                    "take_profit_pct": "take_profit_percentage"
                },
                "position_sizer": {
                    "BASE_POSITION_SIZE": "base_fraction",
                    "MIN_POSITION_SIZE_RATIO": "min_fraction",
                    "MAX_POSITION_SIZE_RATIO": "max_fraction",
                    "CONFIDENCE_MULTIPLIER": "confidence_multiplier"
                }
            },
            component_configs={
                "signal_generator": {"name": "ml_basic_signals"},
                "risk_manager": {"name": "fixed_risk"},
                "position_sizer": {"name": "confidence_weighted"}
            }
        )

        # ML Adaptive Strategy mapping
        mappings["MlAdaptive"] = ComponentMapping(
            signal_generator_type=MLSignalGenerator,
            risk_manager_type=RegimeAdaptiveRiskManager,
            position_sizer_type=ConfidenceWeightedSizer,
            parameter_mappings={
                "signal_generator": {
                    "model_path": "model_path",
                    "sequence_length": "sequence_length",
                    "use_prediction_engine": "use_prediction_engine",
                    "model_name": "model_name"
                },
                "risk_manager": {
                    "stop_loss_pct": "base_stop_loss_percentage",
                    "take_profit_pct": "base_take_profit_percentage"
                },
                "position_sizer": {
                    "BASE_POSITION_SIZE": "base_fraction",
                    "MIN_POSITION_SIZE_RATIO": "min_fraction",
                    "MAX_POSITION_SIZE_RATIO": "max_fraction",
                    "CONFIDENCE_MULTIPLIER": "confidence_multiplier"
                }
            },
            component_configs={
                "signal_generator": {"name": "ml_adaptive_signals"},
                "risk_manager": {"name": "regime_adaptive_risk"},
                "position_sizer": {"name": "confidence_weighted"}
            }
        )

        # Generic technical strategy mapping (fallback)
        mappings["BaseStrategy"] = ComponentMapping(
            signal_generator_type=TechnicalSignalGenerator,
            risk_manager_type=VolatilityRiskManager,
            position_sizer_type=FixedFractionSizer,
            parameter_mappings={
                "signal_generator": {},
                "risk_manager": {},
                "position_sizer": {}
            },
            component_configs={
                "signal_generator": {"name": "technical_signals"},
                "risk_manager": {},  # VolatilityRiskManager doesn't accept name parameter
                "position_sizer": {}  # FixedFractionSizer doesn't accept name parameter
            }
        )

        return mappings

    def _get_default_mapping(self, strategy: BaseStrategy) -> Optional[ComponentMapping]:
        """Get default component mapping for a strategy"""
        strategy_type = strategy.__class__.__name__

        # Try exact match first
        if strategy_type in self._conversion_mappings:
            return self._conversion_mappings[strategy_type]

        # Try base class matches
        for base_class in strategy.__class__.__mro__[1:]:  # Skip the class itself
            if base_class.__name__ in self._conversion_mappings:
                self.logger.info(f"Using mapping for base class {base_class.__name__} for {strategy_type}")
                return self._conversion_mappings[base_class.__name__]

        # Fallback to generic mapping
        if "BaseStrategy" in self._conversion_mappings:
            self.logger.warning(f"Using generic BaseStrategy mapping for {strategy_type}")
            return self._conversion_mappings["BaseStrategy"]

        return None

    def _extract_legacy_parameters(self, strategy: BaseStrategy) -> Dict[str, Any]:
        """Extract parameters from legacy strategy"""
        params = {}

        try:
            # Get strategy parameters if available
            if hasattr(strategy, "get_parameters"):
                strategy_params = strategy.get_parameters()
                if isinstance(strategy_params, dict):
                    params.update(strategy_params)

            # Extract common attributes
            common_attrs = [
                "model_path", "sequence_length", "stop_loss_pct", "take_profit_pct",
                "use_prediction_engine", "model_name", "trading_pair"
            ]

            for attr in common_attrs:
                if hasattr(strategy, attr):
                    params[attr] = getattr(strategy, attr)

            # Extract class constants
            class_constants = [
                "BASE_POSITION_SIZE", "MIN_POSITION_SIZE_RATIO", "MAX_POSITION_SIZE_RATIO",
                "CONFIDENCE_MULTIPLIER", "SHORT_ENTRY_THRESHOLD"
            ]

            for const in class_constants:
                if hasattr(strategy.__class__, const):
                    params[const] = getattr(strategy.__class__, const)

        except Exception as e:
            self.logger.warning(f"Error extracting parameters from {strategy.__class__.__name__}: {e}")

        return params

    def _map_parameters(self, legacy_params: Dict[str, Any],
                       mapping: ComponentMapping) -> Dict[str, Dict[str, Any]]:
        """Map legacy parameters to component parameters"""
        component_params = {
            "signal_generator": {},
            "risk_manager": {},
            "position_sizer": {}
        }

        # Apply parameter mappings
        for component, param_mapping in mapping.parameter_mappings.items():
            for legacy_param, component_param in param_mapping.items():
                if legacy_param in legacy_params:
                    component_params[component][component_param] = legacy_params[legacy_param]

        # Add component-specific configurations
        for component, config in mapping.component_configs.items():
            component_params[component].update(config)

        return component_params

    def _create_components(self, mapping: ComponentMapping,
                          component_params: Dict[str, Dict[str, Any]],
                          report: ConversionReport) -> Dict[str, Optional[Union[SignalGenerator, RiskManager, PositionSizer]]]:
        """Create component instances"""
        components = {}

        try:
            # Create signal generator
            signal_params = component_params.get("signal_generator", {})
            components["signal_generator"] = mapping.signal_generator_type(**signal_params)
            report.audit_trail.append(f"Created signal generator: {mapping.signal_generator_type.__name__}")

        except Exception as e:
            error_msg = f"Failed to create signal generator: {e}"
            report.errors.append(error_msg)
            components["signal_generator"] = None
            self.logger.error(error_msg)

        try:
            # Create risk manager
            risk_params = component_params.get("risk_manager", {})
            components["risk_manager"] = mapping.risk_manager_type(**risk_params)
            report.audit_trail.append(f"Created risk manager: {mapping.risk_manager_type.__name__}")

        except Exception as e:
            error_msg = f"Failed to create risk manager: {e}"
            report.errors.append(error_msg)
            components["risk_manager"] = None
            self.logger.error(error_msg)

        try:
            # Create position sizer
            sizer_params = component_params.get("position_sizer", {})
            components["position_sizer"] = mapping.position_sizer_type(**sizer_params)
            report.audit_trail.append(f"Created position sizer: {mapping.position_sizer_type.__name__}")

        except Exception as e:
            error_msg = f"Failed to create position sizer: {e}"
            report.errors.append(error_msg)
            components["position_sizer"] = None
            self.logger.error(error_msg)

        return components

    def _validate_conversion(self, legacy_strategy: BaseStrategy,
                           adapter: LegacyStrategyAdapter,
                           report: ConversionReport) -> Dict[str, bool]:
        """Validate the conversion by comparing basic functionality"""
        validation_results = {}

        try:
            # Test 1: Parameter extraction
            legacy_params = legacy_strategy.get_parameters() if hasattr(legacy_strategy, "get_parameters") else {}
            adapter_params = adapter.get_parameters()

            validation_results["parameters_extracted"] = len(legacy_params) > 0 or len(adapter_params) > 0
            report.audit_trail.append(f"Parameter validation: {validation_results['parameters_extracted']}")

            # Test 2: Trading pair consistency
            legacy_pair = legacy_strategy.get_trading_pair()
            adapter_pair = adapter.get_trading_pair()
            validation_results["trading_pair_consistent"] = legacy_pair == adapter_pair
            report.audit_trail.append(f"Trading pair validation: {validation_results['trading_pair_consistent']}")

            # Test 3: Component creation
            component_status = adapter.get_component_status()
            validation_results["components_created"] = all(
                "none" not in status.lower() for status in component_status.values()
            )
            report.audit_trail.append(f"Component creation validation: {validation_results['components_created']}")

            # Test 4: Basic interface compatibility
            validation_results["interface_compatible"] = (
                hasattr(adapter, "check_entry_conditions") and
                hasattr(adapter, "check_exit_conditions") and
                hasattr(adapter, "calculate_position_size") and
                hasattr(adapter, "calculate_stop_loss")
            )
            report.audit_trail.append(f"Interface compatibility validation: {validation_results['interface_compatible']}")

        except Exception as e:
            error_msg = f"Validation failed: {e}"
            report.errors.append(error_msg)
            self.logger.error(error_msg)

            # Set all validations to False on error
            validation_results = {
                "parameters_extracted": False,
                "trading_pair_consistent": False,
                "components_created": False,
                "interface_compatible": False
            }

        return validation_results

    def add_custom_mapping(self, strategy_type: str, mapping: ComponentMapping) -> None:
        """
        Add a custom conversion mapping for a strategy type
        
        Args:
            strategy_type: Name of the strategy class
            mapping: Component mapping configuration
        """
        self._conversion_mappings[strategy_type] = mapping
        self.logger.info(f"Added custom mapping for strategy type: {strategy_type}")

    def get_mapping_for_strategy(self, strategy_type: str) -> Optional[ComponentMapping]:
        """
        Get the conversion mapping for a strategy type
        
        Args:
            strategy_type: Name of the strategy class
            
        Returns:
            ComponentMapping if available, None otherwise
        """
        return self._conversion_mappings.get(strategy_type)

    def generate_conversion_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all conversions performed
        
        Returns:
            Dictionary with conversion statistics and summaries
        """
        if not self.conversion_history:
            return {
                "total_conversions": 0,
                "successful_conversions": 0,
                "failed_conversions": 0,
                "success_rate": 0.0,
                "strategy_types_converted": [],
                "common_errors": [],
                "common_warnings": []
            }

        total = len(self.conversion_history)
        successful = sum(1 for report in self.conversion_history if report.success)
        failed = total - successful

        # Collect strategy types
        strategy_types = list(set(report.source_strategy_type for report in self.conversion_history))

        # Collect common errors and warnings
        all_errors = []
        all_warnings = []
        for report in self.conversion_history:
            all_errors.extend(report.errors)
            all_warnings.extend(report.warnings)

        # Count frequency of errors and warnings
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1

        warning_counts = {}
        for warning in all_warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1

        # Get most common errors and warnings
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        common_warnings = sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_conversions": total,
            "successful_conversions": successful,
            "failed_conversions": failed,
            "success_rate": (successful / total) * 100 if total > 0 else 0.0,
            "strategy_types_converted": strategy_types,
            "common_errors": [{"error": error, "count": count} for error, count in common_errors],
            "common_warnings": [{"warning": warning, "count": count} for warning, count in common_warnings],
            "conversion_timeline": [
                {
                    "timestamp": report.conversion_timestamp.isoformat(),
                    "strategy_name": report.strategy_name,
                    "success": report.success
                }
                for report in self.conversion_history
            ]
        }
