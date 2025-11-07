"""Utilities for selecting ONNX Runtime execution providers."""

from __future__ import annotations

import argparse
import logging
import platform
from pathlib import Path

import onnxruntime as ort

logger = logging.getLogger(__name__)

# Provider priorities by platform
_MAC_GPU_PROVIDERS = ["CoreMLExecutionProvider", "MPSExecutionProvider"]
_WINDOWS_GPU_PROVIDERS = ["DmlExecutionProvider"]
_UNIX_GPU_PROVIDERS = ["CUDAExecutionProvider", "ROCMExecutionProvider"]


def get_preferred_providers() -> list[str]:
    """Return the preferred ONNX Runtime providers for the current host.

    The function prioritizes GPU-capable providers when available, including
    CoreML on Apple Silicon Macs. It always falls back to the CPU execution
    provider to ensure inference can proceed even if GPU backends are
    unavailable.
    """

    available_providers = list(ort.get_available_providers())
    system_name = platform.system()

    prioritized: list[str] = []
    if system_name == "Darwin":
        prioritized.extend(_MAC_GPU_PROVIDERS)
    elif system_name == "Windows":
        prioritized.extend(_WINDOWS_GPU_PROVIDERS)
    else:
        prioritized.extend(_UNIX_GPU_PROVIDERS)

    prioritized.append("CPUExecutionProvider")

    selected: list[str] = []
    seen: set[str] = set()
    for provider in prioritized:
        if provider in available_providers and provider not in seen:
            selected.append(provider)
            seen.add(provider)

    if not selected:
        # Ensure we always return at least the CPU provider.
        selected.append("CPUExecutionProvider")

    logger.debug("Available ONNX Runtime providers: %s", available_providers)
    logger.info("Using ONNX Runtime providers: %s", selected)
    return selected


def _load_model_for_validation(model_path: Path, providers: list[str]) -> list[str]:
    """Load a model with the supplied providers and return the active providers."""

    session = ort.InferenceSession(str(model_path), providers=providers)
    return session.get_providers()


def _print_provider_report(include_all: bool, model_path: Path | None) -> None:
    """Print diagnostic information about available and selected providers."""

    available = list(ort.get_available_providers())
    selected = get_preferred_providers()

    print("Detected ONNX Runtime providers on this host:\n")
    for provider in available:
        print(f"  - {provider}")

    print("\nProvider priority after applying ai-trading-bot selection logic:\n")
    for provider in selected:
        print(f"  - {provider}")

    if include_all:
        unsupported = [p for p in selected if p not in available]
        if unsupported:
            print("\nThe following preferred providers are not currently available:")
            for provider in unsupported:
                print(f"  - {provider}")

    if model_path is None:
        return

    active = _load_model_for_validation(model_path, selected)
    print("\nSuccessfully initialized the model. ONNX Runtime activated providers:\n")
    for provider in active:
        print(f"  - {provider}")


def main() -> None:
    """Entry point for verifying ONNX Runtime provider selection."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect ONNX Runtime execution providers and validate the "
            "ai-trading-bot provider preference ordering."
        )
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Show preferred providers that are not currently available on this host.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help=(
            "Optional path to an ONNX model. When supplied, the script attempts to "
            "load the model with the preferred providers to confirm GPU activation."
        ),
    )

    args = parser.parse_args()

    model_path = args.model
    if model_path is not None and not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    _print_provider_report(args.include_missing, model_path)


if __name__ == "__main__":
    main()
