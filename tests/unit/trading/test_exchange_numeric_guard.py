"""Static guard: no raw step/tick snapping or raw float serialization on exchange-bound numerics.

`round(x / step) * step` and `math.floor(x / step) * step` leave float artifacts
(`round(1648.82 / 0.01) * 0.01 == 1648.8200000000001`) that Binance rejects with
51077 (quantity) or -1111 (price); `str(float)` renders small values in scientific
notation (-1100). All such snapping must go through `src.trading.precision`.
"""

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCANNED_PACKAGES = (
    "src/data_providers",
    "src/engines/live",
    "src/engines/shared",
    "src/trading",
)
# The one module allowed to implement snapping.
ALLOWED_FILES = {"src/trading/precision.py"}

_SNAP_FUNCS = {"round", "floor", "ceil", "int", "trunc"}
# Order-parameter dicts in the exchange adapters, and the keys that carry a number.
_ORDER_DICT_NAMES = {"order_params", "sl_params", "body", "params"}
_NUMERIC_ORDER_KEYS = {"quantity", "price", "stopPrice", "stop_price", "size", "amount"}
_SERIALIZERS = {"format_quantity", "format_price"}
_ADAPTER_FILES = {
    "src/data_providers/binance_provider.py",
    "src/data_providers/coinbase_provider.py",
}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return None


def _contains_division(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div | ast.FloorDiv)
        for n in ast.walk(node)
    )


def _is_snap_of_division(node: ast.AST) -> bool:
    """True for `round(a / b)`, `math.floor(a / b)`, `int(a / b)` or `a // b`."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv):
        return True
    return _call_name(node) in _SNAP_FUNCS and _contains_division(node)


def find_raw_step_snaps(source: str) -> list[int]:
    """Line numbers of `<snap of a division> * <anything>` expressions."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if _is_snap_of_division(node.left) or _is_snap_of_division(node.right):
                hits.append(node.lineno)
    return sorted(hits)


def _order_dict_assignments(tree: ast.AST):
    """Yield (line, key, value_node) for numeric keys set on order-parameter dicts."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in _ORDER_DICT_NAMES
                    and isinstance(node.value, ast.Dict)
                ):
                    for key, value in zip(node.value.keys, node.value.values, strict=True):
                        if isinstance(key, ast.Constant) and key.value in _NUMERIC_ORDER_KEYS:
                            yield node.lineno, key.value, value
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in _ORDER_DICT_NAMES
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value in _NUMERIC_ORDER_KEYS
                ):
                    yield node.lineno, target.slice.value, node.value
        # body.update({...}) style
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in _ORDER_DICT_NAMES
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            for key, value in zip(node.args[0].keys, node.args[0].values, strict=True):
                if isinstance(key, ast.Constant) and key.value in _NUMERIC_ORDER_KEYS:
                    yield node.lineno, key.value, value


def find_unserialized_order_numerics(source: str) -> list[tuple[int, str]]:
    """Order-dict numeric entries not produced by `format_quantity`/`format_price`.

    A `**self._conform_order_numerics(...)` splat is the sanctioned boundary guard
    and carries no literal numeric key, so it is never flagged.
    """
    hits = []
    for line, key, value in _order_dict_assignments(ast.parse(source)):
        values = [value.body, value.orelse] if isinstance(value, ast.IfExp) else [value]
        for v in values:
            if isinstance(v, ast.Constant) and v.value is None:
                continue
            if _call_name(v) not in _SERIALIZERS:
                hits.append((line, key))
    return hits


def _scanned_files() -> list[Path]:
    files = []
    for package in SCANNED_PACKAGES:
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            if path.relative_to(REPO_ROOT).as_posix() not in ALLOWED_FILES:
                files.append(path)
    return files


def test_no_raw_step_snap_in_exchange_facing_packages():
    offenders = []
    for path in _scanned_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        offenders += [f"{rel}:{line}" for line in find_raw_step_snaps(path.read_text())]
    assert not offenders, (
        "Raw `round(x / step) * step` style snapping found; use "
        "src.trading.precision.snap_to_step (LESSONS 1.1): " + ", ".join(offenders)
    )


def test_adapter_order_params_use_precision_serializers():
    offenders = []
    for rel in sorted(_ADAPTER_FILES):
        source = (REPO_ROOT / rel).read_text()
        offenders += [
            f"{rel}:{line} ({key})" for line, key in find_unserialized_order_numerics(source)
        ]
    assert not offenders, (
        "Exchange order parameters must be serialized with format_quantity/format_price "
        "(or the provider's _conform_order_numerics guard), never raw floats or str(): "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize(
    "snippet",
    [
        "q = round(x / step) * step",
        "q = math.floor(x / step) * step",
        "q = math.floor(x / step + 1e-9) * step",
        "q = step * math.ceil(x / step)",
        "q = int(x / step) * step",
        "q = (x // step) * step",
    ],
)
def test_detector_flags_known_raw_snap_shapes(snippet):
    assert find_raw_step_snaps(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        "q = snap_to_step(x, step)",
        "pct = round(a / b, 4)",
        "n = round(x * 100) / 100",
        "area = width * height",
    ],
)
def test_detector_ignores_unrelated_arithmetic(snippet):
    assert not find_raw_step_snaps(snippet)


def test_serializer_detector_flags_raw_values():
    bad = 'order_params = {"quantity": quantity, "price": str(price)}'
    assert [key for _, key in find_unserialized_order_numerics(bad)] == ["quantity", "price"]
    good = 'order_params = {"quantity": format_quantity(q), "price": format_price(p)}'
    assert not find_unserialized_order_numerics(good)
