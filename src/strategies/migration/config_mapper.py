"""
Configuration Mapping Utilities

This module provides utilities for mapping configuration parameters between
legacy strategies and component-based strategies, including parameter validation
and transformation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


@dataclass
class ParameterMapping:
    """
    Configuration for mapping a single parameter
    
    Attributes:
        source_key: Key in the source configuration
        target_key: Key in the target configuration
        transformer: Optional function to transform the value
        validator: Optional function to validate the transformed value
        default_value: Default value if source key is missing
        required: Whether this parameter is required
        description: Description of the parameter mapping
    """
    source_key: str
    target_key: str
    transformer: Optional[Callable[[Any], Any]] = None
    validator: Optional[Callable[[Any], bool]] = None
    default_value: Any = None
    required: bool = False
    description: str = ""


@dataclass
class ComponentConfigMapping:
    """
    Configuration mapping for a single component
    
    Attributes:
        component_name: Name of the component (signal_generator, risk_manager, position_sizer)
        parameter_mappings: List of parameter mappings for this component
        static_config: Static configuration values to always include
        validation_rules: Additional validation rules for the component
    """
    component_name: str
    parameter_mappings: List[ParameterMapping]
    static_config: Dict[str, Any]
    validation_rules: List[Callable[[Dict[str, Any]], bool]]


class ConfigMapper:
    """
    Utility class for mapping configuration parameters between legacy and component-based strategies
    
    This class handles parameter transformation, validation, and provides detailed
    mapping reports for audit purposes.
    """

    def __init__(self):
        """Initialize the configuration mapper"""
        self.logger = logging.getLogger("ConfigMapper")

        # Initialize built-in transformers and validators
        self._transformers = self._initialize_transformers()
        self._validators = self._initialize_validators()

        # Track mapping operations
        self.mapping_history: List[Dict[str, Any]] = []

    def map_configuration(self, source_config: Dict[str, Any],
                         component_mappings: List[ComponentConfigMapping]) -> Dict[str, Dict[str, Any]]:
        """
        Map source configuration to component configurations
        
        Args:
            source_config: Source configuration dictionary
            component_mappings: List of component mapping configurations
            
        Returns:
            Dictionary with component configurations
        """
        result = {}
        mapping_report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "source_keys": list(source_config.keys()),
            "mapped_parameters": {},
            "warnings": [],
            "errors": []
        }

        try:
            for component_mapping in component_mappings:
                component_config = self._map_component_config(
                    source_config, component_mapping, mapping_report
                )
                result[component_mapping.component_name] = component_config

            # Store mapping history
            self.mapping_history.append(mapping_report)

            self.logger.info(f"Successfully mapped configuration for {len(component_mappings)} components")

        except Exception as e:
            error_msg = f"Configuration mapping failed: {e}"
            mapping_report["errors"].append(error_msg)
            self.logger.error(error_msg, exc_info=True)
            raise

        return result

    def _map_component_config(self, source_config: Dict[str, Any],
                            component_mapping: ComponentConfigMapping,
                            mapping_report: Dict[str, Any]) -> Dict[str, Any]:
        """Map configuration for a single component"""
        component_config = component_mapping.static_config.copy()
        component_name = component_mapping.component_name

        mapping_report["mapped_parameters"][component_name] = {
            "mapped": [],
            "missing_required": [],
            "validation_failures": [],
            "transformations_applied": []
        }

        # Process each parameter mapping
        for param_mapping in component_mapping.parameter_mappings:
            try:
                self._map_single_parameter(
                    source_config, param_mapping, component_config,
                    mapping_report["mapped_parameters"][component_name]
                )
            except Exception as e:
                error_msg = f"Failed to map parameter {param_mapping.source_key}: {e}"
                mapping_report["errors"].append(error_msg)
                self.logger.error(error_msg)

        # Apply component validation rules
        for validation_rule in component_mapping.validation_rules:
            try:
                if not validation_rule(component_config):
                    warning_msg = f"Validation rule failed for component {component_name}"
                    mapping_report["warnings"].append(warning_msg)
                    self.logger.warning(warning_msg)
            except Exception as e:
                error_msg = f"Validation rule error for component {component_name}: {e}"
                mapping_report["errors"].append(error_msg)
                self.logger.error(error_msg)

        return component_config

    def _map_single_parameter(self, source_config: Dict[str, Any],
                            param_mapping: ParameterMapping,
                            target_config: Dict[str, Any],
                            component_report: Dict[str, List[str]]) -> None:
        """Map a single parameter"""
        source_key = param_mapping.source_key
        target_key = param_mapping.target_key

        # Check if source key exists
        if source_key not in source_config:
            if param_mapping.required:
                component_report["missing_required"].append(source_key)
                self.logger.warning(f"Required parameter {source_key} not found in source config")

            # Use default value if available
            if param_mapping.default_value is not None:
                target_config[target_key] = param_mapping.default_value
                component_report["mapped"].append(f"{source_key} -> {target_key} (default)")

            return

        # Get source value
        source_value = source_config[source_key]

        # Apply transformer if specified
        if param_mapping.transformer:
            try:
                transformed_value = param_mapping.transformer(source_value)
                component_report["transformations_applied"].append(
                    f"{source_key}: {source_value} -> {transformed_value}"
                )
            except Exception as e:
                error_msg = f"Transformation failed for {source_key}: {e}"
                self.logger.error(error_msg)
                transformed_value = source_value
        else:
            transformed_value = source_value

        # Apply validator if specified
        if param_mapping.validator:
            try:
                if not param_mapping.validator(transformed_value):
                    component_report["validation_failures"].append(
                        f"{source_key}: {transformed_value} failed validation"
                    )
                    self.logger.warning(f"Validation failed for {source_key}: {transformed_value}")
                    return
            except Exception as e:
                error_msg = f"Validation error for {source_key}: {e}"
                self.logger.error(error_msg)

        # Set target value
        target_config[target_key] = transformed_value
        component_report["mapped"].append(f"{source_key} -> {target_key}")

    def _initialize_transformers(self) -> Dict[str, Callable[[Any], Any]]:
        """Initialize built-in parameter transformers"""
        return {
            "percentage_to_decimal": lambda x: float(x) / 100.0 if isinstance(x, (int, float)) else x,
            "decimal_to_percentage": lambda x: float(x) * 100.0 if isinstance(x, (int, float)) else x,
            "string_to_float": lambda x: float(x) if isinstance(x, str) and x.replace(".", "").isdigit() else x,
            "string_to_int": lambda x: int(x) if isinstance(x, str) and x.isdigit() else x,
            "bool_to_string": lambda x: str(x).lower() if isinstance(x, bool) else x,
            "string_to_bool": lambda x: x.lower() in ("true", "1", "yes", "on") if isinstance(x, str) else bool(x),
            "clamp_0_1": lambda x: max(0.0, min(1.0, float(x))) if isinstance(x, (int, float)) else x,
            "abs_value": lambda x: abs(x) if isinstance(x, (int, float)) else x,
            "ensure_positive": lambda x: max(0.001, float(x)) if isinstance(x, (int, float)) else x,
            "round_to_decimals": lambda decimals: lambda x: round(float(x), decimals) if isinstance(x, (int, float)) else x
        }

    def _initialize_validators(self) -> Dict[str, Callable[[Any], bool]]:
        """Initialize built-in parameter validators"""
        return {
            "is_positive": lambda x: isinstance(x, (int, float)) and x > 0,
            "is_non_negative": lambda x: isinstance(x, (int, float)) and x >= 0,
            "is_percentage": lambda x: isinstance(x, (int, float)) and 0 <= x <= 100,
            "is_decimal_percentage": lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
            "is_integer": lambda x: isinstance(x, int) or (isinstance(x, float) and x.is_integer()),
            "is_string": lambda x: isinstance(x, str),
            "is_boolean": lambda x: isinstance(x, bool),
            "is_not_empty": lambda x: x is not None and (not isinstance(x, str) or len(x.strip()) > 0),
            "is_finite": lambda x: isinstance(x, (int, float)) and np.isfinite(x),
            "range_validator": lambda min_val, max_val: lambda x: isinstance(x, (int, float)) and min_val <= x <= max_val
        }

    def get_transformer(self, name: str) -> Optional[Callable[[Any], Any]]:
        """Get a built-in transformer by name"""
        return self._transformers.get(name)

    def get_validator(self, name: str) -> Optional[Callable[[Any], bool]]:
        """Get a built-in validator by name"""
        return self._validators.get(name)

    def add_custom_transformer(self, name: str, transformer: Callable[[Any], Any]) -> None:
        """Add a custom transformer"""
        self._transformers[name] = transformer
        self.logger.info(f"Added custom transformer: {name}")

    def add_custom_validator(self, name: str, validator: Callable[[Any], bool]) -> None:
        """Add a custom validator"""
        self._validators[name] = validator
        self.logger.info(f"Added custom validator: {name}")

    def create_ml_basic_mappings(self) -> List[ComponentConfigMapping]:
        """Create parameter mappings for ML Basic strategy"""
        return [
            # Signal Generator mappings
            ComponentConfigMapping(
                component_name="signal_generator",
                parameter_mappings=[
                    ParameterMapping(
                        source_key="model_path",
                        target_key="model_path",
                        validator=self._validators["is_string"],
                        required=True,
                        description="Path to the ONNX model file"
                    ),
                    ParameterMapping(
                        source_key="sequence_length",
                        target_key="sequence_length",
                        validator=self._validators["is_positive"],
                        default_value=120,
                        description="Sequence length for ML model input"
                    ),
                    ParameterMapping(
                        source_key="use_prediction_engine",
                        target_key="use_prediction_engine",
                        validator=self._validators["is_boolean"],
                        default_value=False,
                        description="Whether to use the prediction engine"
                    ),
                    ParameterMapping(
                        source_key="model_name",
                        target_key="model_name",
                        validator=self._validators["is_string"],
                        description="Name of the model in the prediction engine"
                    )
                ],
                static_config={"name": "ml_basic_signals"},
                validation_rules=[
                    lambda config: config.get("sequence_length", 0) > 0,
                    lambda config: isinstance(config.get("model_path"), str) and len(config["model_path"]) > 0
                ]
            ),

            # Risk Manager mappings
            ComponentConfigMapping(
                component_name="risk_manager",
                parameter_mappings=[
                    ParameterMapping(
                        source_key="stop_loss_pct",
                        target_key="stop_loss_percentage",
                        validator=self._validators["is_decimal_percentage"],
                        default_value=0.02,
                        description="Stop loss percentage as decimal"
                    ),
                    ParameterMapping(
                        source_key="take_profit_pct",
                        target_key="take_profit_percentage",
                        validator=self._validators["is_decimal_percentage"],
                        default_value=0.04,
                        description="Take profit percentage as decimal"
                    )
                ],
                static_config={"name": "fixed_risk"},
                validation_rules=[
                    lambda config: config.get("stop_loss_percentage", 0) > 0,
                    lambda config: config.get("take_profit_percentage", 0) > config.get("stop_loss_percentage", 0)
                ]
            ),

            # Position Sizer mappings
            ComponentConfigMapping(
                component_name="position_sizer",
                parameter_mappings=[
                    ParameterMapping(
                        source_key="BASE_POSITION_SIZE",
                        target_key="base_fraction",
                        validator=self._validators["is_decimal_percentage"],
                        default_value=0.2,
                        description="Base position size as fraction of balance"
                    ),
                    ParameterMapping(
                        source_key="MIN_POSITION_SIZE_RATIO",
                        target_key="min_fraction",
                        validator=self._validators["is_decimal_percentage"],
                        default_value=0.05,
                        description="Minimum position size as fraction of balance"
                    ),
                    ParameterMapping(
                        source_key="MAX_POSITION_SIZE_RATIO",
                        target_key="max_fraction",
                        validator=self._validators["is_decimal_percentage"],
                        default_value=0.25,
                        description="Maximum position size as fraction of balance"
                    ),
                    ParameterMapping(
                        source_key="CONFIDENCE_MULTIPLIER",
                        target_key="confidence_multiplier",
                        validator=self._validators["is_positive"],
                        default_value=12,
                        description="Multiplier for confidence-based position sizing"
                    )
                ],
                static_config={"name": "confidence_weighted"},
                validation_rules=[
                    lambda config: config.get("min_fraction", 0) <= config.get("base_fraction", 0) <= config.get("max_fraction", 1),
                    lambda config: config.get("confidence_multiplier", 0) > 0
                ]
            )
        ]

    def create_ml_adaptive_mappings(self) -> List[ComponentConfigMapping]:
        """Create parameter mappings for ML Adaptive strategy"""
        return [
            # Signal Generator mappings (similar to ML Basic but with adaptive features)
            ComponentConfigMapping(
                component_name="signal_generator",
                parameter_mappings=[
                    ParameterMapping(
                        source_key="model_path",
                        target_key="model_path",
                        validator=self._validators["is_string"],
                        required=True,
                        description="Path to the ONNX model file"
                    ),
                    ParameterMapping(
                        source_key="sequence_length",
                        target_key="sequence_length",
                        validator=self._validators["is_positive"],
                        default_value=120,
                        description="Sequence length for ML model input"
                    ),
                    ParameterMapping(
                        source_key="use_prediction_engine",
                        target_key="use_prediction_engine",
                        validator=self._validators["is_boolean"],
                        default_value=False,
                        description="Whether to use the prediction engine"
                    ),
                    ParameterMapping(
                        source_key="model_name",
                        target_key="model_name",
                        validator=self._validators["is_string"],
                        description="Name of the model in the prediction engine"
                    ),
                    ParameterMapping(
                        source_key="SHORT_ENTRY_THRESHOLD",
                        target_key="short_entry_threshold",
                        validator=lambda x: isinstance(x, (int, float)) and x < 0,
                        default_value=-0.0005,
                        description="Threshold for short entry signals"
                    )
                ],
                static_config={"name": "ml_adaptive_signals", "regime_aware": True},
                validation_rules=[
                    lambda config: config.get("sequence_length", 0) > 0,
                    lambda config: isinstance(config.get("model_path"), str) and len(config["model_path"]) > 0,
                    lambda config: config.get("short_entry_threshold", 0) < 0
                ]
            ),

            # Risk Manager mappings (regime-adaptive)
            ComponentConfigMapping(
                component_name="risk_manager",
                parameter_mappings=[
                    ParameterMapping(
                        source_key="stop_loss_pct",
                        target_key="base_stop_loss_percentage",
                        validator=self._validators["is_decimal_percentage"],
                        default_value=0.02,
                        description="Base stop loss percentage as decimal"
                    ),
                    ParameterMapping(
                        source_key="take_profit_pct",
                        target_key="base_take_profit_percentage",
                        validator=self._validators["is_decimal_percentage"],
                        default_value=0.04,
                        description="Base take profit percentage as decimal"
                    )
                ],
                static_config={
                    "name": "regime_adaptive_risk",
                    "regime_multipliers": {
                        "bull_low_vol": {"stop_loss": 0.8, "take_profit": 1.2},
                        "bull_high_vol": {"stop_loss": 1.2, "take_profit": 1.0},
                        "bear_low_vol": {"stop_loss": 1.0, "take_profit": 0.8},
                        "bear_high_vol": {"stop_loss": 1.5, "take_profit": 0.6},
                        "range_low_vol": {"stop_loss": 0.9, "take_profit": 1.1},
                        "range_high_vol": {"stop_loss": 1.3, "take_profit": 0.9}
                    }
                },
                validation_rules=[
                    lambda config: config.get("base_stop_loss_percentage", 0) > 0,
                    lambda config: config.get("base_take_profit_percentage", 0) > config.get("base_stop_loss_percentage", 0)
                ]
            ),

            # Position Sizer mappings (same as ML Basic)
            ComponentConfigMapping(
                component_name="position_sizer",
                parameter_mappings=[
                    ParameterMapping(
                        source_key="BASE_POSITION_SIZE",
                        target_key="base_fraction",
                        validator=self._validators["is_decimal_percentage"],
                        default_value=0.2,
                        description="Base position size as fraction of balance"
                    ),
                    ParameterMapping(
                        source_key="MIN_POSITION_SIZE_RATIO",
                        target_key="min_fraction",
                        validator=self._validators["is_decimal_percentage"],
                        default_value=0.05,
                        description="Minimum position size as fraction of balance"
                    ),
                    ParameterMapping(
                        source_key="MAX_POSITION_SIZE_RATIO",
                        target_key="max_fraction",
                        validator=self._validators["is_decimal_percentage"],
                        default_value=0.25,
                        description="Maximum position size as fraction of balance"
                    ),
                    ParameterMapping(
                        source_key="CONFIDENCE_MULTIPLIER",
                        target_key="confidence_multiplier",
                        validator=self._validators["is_positive"],
                        default_value=12,
                        description="Multiplier for confidence-based position sizing"
                    )
                ],
                static_config={"name": "confidence_weighted"},
                validation_rules=[
                    lambda config: config.get("min_fraction", 0) <= config.get("base_fraction", 0) <= config.get("max_fraction", 1),
                    lambda config: config.get("confidence_multiplier", 0) > 0
                ]
            )
        ]

    def get_mapping_history(self) -> List[Dict[str, Any]]:
        """Get history of all mapping operations"""
        return self.mapping_history.copy()

    def clear_mapping_history(self) -> None:
        """Clear the mapping history"""
        self.mapping_history.clear()
        self.logger.info("Mapping history cleared")

    def validate_component_config(self, component_config: Dict[str, Any],
                                component_mapping: ComponentConfigMapping) -> Dict[str, Any]:
        """
        Validate a component configuration against its mapping rules
        
        Args:
            component_config: Configuration to validate
            component_mapping: Mapping configuration with validation rules
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_required": [],
            "validation_details": {}
        }

        try:
            # Check required parameters
            for param_mapping in component_mapping.parameter_mappings:
                if param_mapping.required and param_mapping.target_key not in component_config:
                    validation_results["missing_required"].append(param_mapping.target_key)
                    validation_results["valid"] = False

            # Apply validation rules
            for i, validation_rule in enumerate(component_mapping.validation_rules):
                try:
                    if not validation_rule(component_config):
                        error_msg = f"Validation rule {i+1} failed"
                        validation_results["errors"].append(error_msg)
                        validation_results["valid"] = False
                except Exception as e:
                    error_msg = f"Validation rule {i+1} error: {e}"
                    validation_results["errors"].append(error_msg)
                    validation_results["valid"] = False

            # Validate individual parameters
            for param_mapping in component_mapping.parameter_mappings:
                target_key = param_mapping.target_key
                if target_key in component_config and param_mapping.validator:
                    try:
                        value = component_config[target_key]
                        if not param_mapping.validator(value):
                            error_msg = f"Parameter {target_key} failed validation: {value}"
                            validation_results["errors"].append(error_msg)
                            validation_results["valid"] = False

                        validation_results["validation_details"][target_key] = {
                            "value": value,
                            "valid": param_mapping.validator(value),
                            "description": param_mapping.description
                        }
                    except Exception as e:
                        error_msg = f"Parameter {target_key} validation error: {e}"
                        validation_results["errors"].append(error_msg)
                        validation_results["valid"] = False

        except Exception as e:
            validation_results["errors"].append(f"Validation process error: {e}")
            validation_results["valid"] = False
            self.logger.error(f"Component config validation failed: {e}", exc_info=True)

        return validation_results
