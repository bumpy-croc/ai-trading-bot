"""Cloud training provider registry.

Provides factory functions for loading and instantiating cloud training providers.
Follows the same pattern as src/config/providers for extensibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ml.cloud.exceptions import ProviderNotAvailableError

if TYPE_CHECKING:
    from src.ml.cloud.providers.base import CloudTrainingProvider


def get_provider(provider_name: str) -> CloudTrainingProvider:
    """Get cloud training provider by name.

    Factory function that returns an initialized provider instance.
    Supports lazy loading to avoid importing boto3 when not needed.

    Args:
        provider_name: Provider identifier (sagemaker, local)

    Returns:
        Initialized CloudTrainingProvider instance

    Raises:
        ProviderNotAvailableError: If provider not found or not configured
    """
    # Lazy import to avoid boto3 dependency when not using cloud training
    if provider_name == "sagemaker":
        from src.ml.cloud.providers.sagemaker import SageMakerProvider

        return SageMakerProvider()

    if provider_name == "local":
        from src.ml.cloud.providers.local import LocalProvider

        return LocalProvider()

    available = list_providers()
    raise ProviderNotAvailableError(
        f"Provider '{provider_name}' not found. Available providers: {', '.join(available)}"
    )


def list_providers() -> list[str]:
    """List all available provider names.

    Returns:
        List of provider identifiers
    """
    return ["sagemaker", "local"]
