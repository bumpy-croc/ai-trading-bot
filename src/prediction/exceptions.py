"""
Exception classes for the prediction engine.
"""


class PredictionEngineError(Exception):
    """Base exception for prediction engine errors"""

    pass


class InvalidInputError(PredictionEngineError):
    """Raised when input data is invalid"""

    pass


class ModelNotFoundError(PredictionEngineError):
    """Raised when requested model is not available"""

    pass


class FeatureExtractionError(PredictionEngineError):
    """Raised when feature extraction fails"""

    pass


class PredictionTimeoutError(PredictionEngineError):
    """Raised when prediction takes too long"""

    pass
