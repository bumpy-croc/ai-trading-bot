from __future__ import annotations

import inspect

import pytest


pytestmark = pytest.mark.integration


def _is_signature_compatible(real_sig: inspect.Signature, mock_sig: inspect.Signature) -> bool:
    """Return True if mock signature is compatible with real signature.

    Compatibility rules:
    - All real positional-or-keyword parameter names must exist in mock OR mock must accept **kwargs
    - Default values and annotations are not strictly enforced (allow flexibility in tests)
    - Return annotation is ignored
    """
    real_params = real_sig.parameters
    mock_params = mock_sig.parameters

    # Fast-path: mock has **kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in mock_params.values()):
        return True

    mock_param_names = set(mock_params.keys())
    for name, param in real_params.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            # Skip varargs in real signature
            continue
        if name not in mock_param_names:
            return False
    return True


@pytest.mark.parametrize(
    "method_name",
    [
        "create_trading_session",
        "end_trading_session",
        "log_position",
        "update_position",
        "log_trade",
        "close_position",
        "log_event",
        "log_account_snapshot",
        "update_balance",
        "get_current_balance",
        # Commonly used helpers
        "update_balance",
        "get_current_balance",
        "log_position",
        "log_event",
        "log_account_snapshot",
    ],
)
def test_mock_and_real_database_signatures_are_compatible(method_name: str):
    # Import here to avoid test collection issues in environments without all deps
    from database.manager import DatabaseManager  # type: ignore
    from tests.mocks.mock_database import MockDatabaseManager  # type: ignore

    assert hasattr(DatabaseManager, method_name), f"DatabaseManager missing {method_name}"
    assert hasattr(MockDatabaseManager, method_name), f"MockDatabaseManager missing {method_name}"

    real_method = getattr(DatabaseManager, method_name)
    mock_method = getattr(MockDatabaseManager, method_name)

    real_sig = inspect.signature(real_method)
    mock_sig = inspect.signature(mock_method)

    assert _is_signature_compatible(
        real_sig, mock_sig
    ), f"Signature mismatch for {method_name}: real={real_sig} mock={mock_sig}"


