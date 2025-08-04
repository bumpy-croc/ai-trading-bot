"""
Prediction engine exception classes.

This module defines custom exception classes for the prediction engine
to provide clear error handling and debugging information.
"""


class PredictionEngineError(Exception):
    """Base exception for prediction engine errors."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class InvalidInputError(PredictionEngineError):
    """Raised when input data is invalid."""
    
    def __init__(self, message: str, invalid_columns: list = None):
        super().__init__(message, error_code="INVALID_INPUT")
        self.invalid_columns = invalid_columns or []


class ModelNotFoundError(PredictionEngineError):
    """Raised when requested model is not available."""
    
    def __init__(self, model_name: str, available_models: list = None):
        message = f"Model '{model_name}' not found"
        if available_models:
            message += f". Available models: {', '.join(available_models)}"
        super().__init__(message, error_code="MODEL_NOT_FOUND")
        self.model_name = model_name
        self.available_models = available_models or []


class ModelLoadError(PredictionEngineError):
    """Raised when model fails to load."""
    
    def __init__(self, model_name: str, original_error: str = None):
        message = f"Failed to load model '{model_name}'"
        if original_error:
            message += f": {original_error}"
        super().__init__(message, error_code="MODEL_LOAD_ERROR")
        self.model_name = model_name
        self.original_error = original_error


class FeatureExtractionError(PredictionEngineError):
    """Raised when feature extraction fails."""
    
    def __init__(self, message: str, feature_stage: str = None):
        super().__init__(message, error_code="FEATURE_EXTRACTION_ERROR")
        self.feature_stage = feature_stage


class PredictionTimeoutError(PredictionEngineError):
    """Raised when prediction takes too long."""
    
    def __init__(self, timeout_seconds: float, actual_time: float = None):
        message = f"Prediction timed out after {timeout_seconds} seconds"
        if actual_time:
            message += f" (actual time: {actual_time:.2f}s)"
        super().__init__(message, error_code="PREDICTION_TIMEOUT")
        self.timeout_seconds = timeout_seconds
        self.actual_time = actual_time


class InsufficientDataError(PredictionEngineError):
    """Raised when there is insufficient data for prediction."""
    
    def __init__(self, required_rows: int, actual_rows: int, data_type: str = "data"):
        message = f"Insufficient {data_type}: {actual_rows} rows provided, {required_rows} required"
        super().__init__(message, error_code="INSUFFICIENT_DATA")
        self.required_rows = required_rows
        self.actual_rows = actual_rows
        self.data_type = data_type


class ConfigurationError(PredictionEngineError):
    """Raised when there is a configuration error."""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, error_code="CONFIGURATION_ERROR")
        self.config_key = config_key


class ModelInferenceError(PredictionEngineError):
    """Raised when model inference fails."""
    
    def __init__(self, model_name: str, original_error: str = None):
        message = f"Model inference failed for '{model_name}'"
        if original_error:
            message += f": {original_error}"
        super().__init__(message, error_code="MODEL_INFERENCE_ERROR")
        self.model_name = model_name
        self.original_error = original_error


class CacheError(PredictionEngineError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, cache_type: str = None):
        super().__init__(message, error_code="CACHE_ERROR")
        self.cache_type = cache_type


class ValidationError(PredictionEngineError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, validation_type: str = None):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.validation_type = validation_type