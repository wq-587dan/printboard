"""Print output parser for extracting metrics from text.

This module provides regex-based parsing of print statements to extract
numeric metric key-value pairs. It supports multiple common formats
used in deep learning training loops and allows custom regex patterns.

Supported formats:
    - key: value  (colon separator, with or without spaces)
    - key=value   (equals separator, with or without spaces)
    - pipe-separated: "| key: value"
    - dash-separated: "-- key: value"
    - progress bar: "1/10 [====] - loss: 0.5"
    - percentage: "acc: 95%"
    - JSON-like: "{'loss': 0.3456}"
    - scientific notation: "lr: 1.5e-4"
    - special values: "loss: nan", "loss: inf"

Not supported (use custom pattern):
    - Natural language: "Loss is 0.3456"
    - Multi-word keys: "Train Loss: 0.3456" (matches "Loss" only)
    - Leading-dot decimals: ".5" (use "0.5")
"""

from __future__ import annotations

import re
from typing import Optional

# Maximum allowed key length to avoid capturing meaningless long strings.
_MAX_KEY_LENGTH = 32

# Precompiled pattern for key: value format (colon separator).
# Supports optional spaces around the separator: "loss: 0.5", "loss:0.5", "loss : 0.5"
_PATTERNS_KEY_VALUE = re.compile(
    r"(?P<key>[\w]{1,32})\s*[:=]\s*(?P<value>[+-]?[\d]+(?:\.[\d]*)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Precompiled pattern for pipe-separated sections containing key: value pairs.
# Handles formats like: "epoch 3 | loss: 0.3456 | acc: 0.92"
_PATTERNS_PIPE_SEPARATED = re.compile(
    r"\|\s*(?P<key>[\w]{1,32})\s*:\s*(?P<value>[+-]?[\d]+(?:\.[\d]*)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Precompiled pattern for dash-separated sections containing key: value pairs.
# Handles formats like: "Step 100 - loss: 0.3456" or "[====] - loss: 0.5"
_PATTERNS_DASH_SEPARATED = re.compile(
    r"-+\s*(?P<key>[\w]{1,32})\s*:\s*(?P<value>[+-]?[\d]+(?:\.[\d]*)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Precompiled pattern for JSON-like dict output: {'loss': 0.3456, 'acc': 0.92}
# Also handles double quotes and no-quotes formats.
_PATTERNS_JSON_LIKE = re.compile(
    r"""['"]?(?P<key>[\w]{1,32})['"]?\s*:\s*['"]?(?P<value>[+-]?[\d]+(?:\.[\d]*)?(?:[eE][+-]?\d+)?)['"]?""",
    re.IGNORECASE,
)

# Precompiled pattern for percentage values: "acc: 95%"
_PATTERNS_PERCENTAGE = re.compile(
    r"(?P<key>[\w]{1,32})\s*:\s*(?P<value>[+-]?[\d]+(?:\.[\d]*)?)\s*%",
    re.IGNORECASE,
)

# Special float values that are valid but contain no digits.
_SPECIAL_FLOAT_VALUES = {
    "nan",
    "inf",
    "-inf",
    "+inf",
    "infinity",
    "-infinity",
    "+infinity",
}

# Pattern to quickly check if a line contains any numeric value worth parsing.
# Also matches special float keywords (nan, inf).
_HAS_NUMBER = re.compile(r"[\d]+(?:\.[\d]*)?(?:[eE][+-]?\d+)?|nan|inf", re.IGNORECASE)


def _is_valid_key(key: str) -> bool:
    """Check if a key is valid for TensorBoard logging.

    Args:
        key: The key string to validate.

    Returns:
        True if the key is a valid metric name.
    """
    return 0 < len(key) <= _MAX_KEY_LENGTH


def _safe_float(value_str: str) -> Optional[float]:
    """Safely convert a string to float, returning None on failure.

    Handles standard numeric strings and special values (nan, inf).

    Args:
        value_str: The string to convert.

    Returns:
        The float value, or None if conversion fails.
    """
    try:
        return float(value_str)
    except (ValueError, OverflowError):
        return None


def _parse_special_value(text: str) -> dict[str, float]:
    """Parse special float values (nan, inf) from text.

    Checks for patterns like "loss: nan" or "loss: inf" that don't
    contain digits but are valid floating-point values.

    Args:
        text: The text line to check.

    Returns:
        Dict of special value metrics found.
    """
    metrics: dict[str, float] = {}
    text_lower = text.lower()

    for special in _SPECIAL_FLOAT_VALUES:
        # Match "key: special" pattern where special is nan/inf/etc.
        pattern = rf"(?P<key>[\w]{{1,32}})\s*:\s*{re.escape(special)}\b"
        for match in re.finditer(pattern, text_lower):
            key = match.group("key")
            if _is_valid_key(key):
                value = _safe_float(special)
                if value is not None:
                    metrics[key] = value

    return metrics


def parse_print_output(
    text: str,
    custom_pattern: Optional[re.Pattern[str]] = None,
) -> dict[str, float]:
    """Parse a single line of print output into metric key-value pairs.

    This function attempts to extract numeric metrics from text using
    multiple regex strategies. It supports common formats like "loss: 0.5",
    "acc=0.92", pipe-separated, dash-separated, JSON-like, and percentage
    formats. It also handles special float values (nan, inf).

    The parsing follows a priority order:
    1. Custom pattern (if provided) - highest priority, all-or-nothing
    2. Pipe-separated patterns (higher specificity)
    3. Dash-separated patterns
    4. Percentage patterns
    5. JSON-like dict patterns
    6. General key: value / key=value patterns (lowest specificity)
    7. Special values (nan, inf)

    Lines without numeric values are skipped as an optimization.

    Args:
        text: A single line of print output text.
        custom_pattern: Optional custom regex pattern with named capture groups.
                        Group names become metric tags, group values become metric values.

    Returns:
        A dictionary mapping metric names (tags) to their numeric values.
        Duplicate keys are resolved by keeping the last matched value.

    Examples:
        >>> parse_print_output("loss: 0.3456, acc: 0.92")
        {'loss': 0.3456, 'acc': 0.92}

        >>> parse_print_output("epoch 3 | loss: 0.1 | acc: 0.95")
        {'loss': 0.1, 'acc': 0.95}

        >>> parse_print_output("Training started...")
        {}
    """
    if not text or not text.strip():
        return {}

    # Early exit: skip lines with no numeric or special float content.
    if not _HAS_NUMBER.search(text):
        return {}

    metrics: dict[str, float] = {}

    # Strategy 1: Custom pattern (all-or-nothing: user intent overrides built-ins).
    if custom_pattern is not None:
        for match in custom_pattern.finditer(text):
            for key, value_str in match.groupdict().items():
                if value_str is None:
                    continue
                value = _safe_float(value_str)
                if value is not None and _is_valid_key(key):
                    metrics[key] = value
        # Custom pattern is all-or-nothing: if user provides one,
        # don't fall back to built-in heuristics.
        return metrics

    # Strategy 2: Pipe-separated patterns (higher specificity).
    for match in _PATTERNS_PIPE_SEPARATED.finditer(text):
        key = match.group("key")
        value_str = match.group("value")
        value = _safe_float(value_str)
        if value is not None and _is_valid_key(key):
            metrics[key] = value

    # Strategy 3: Dash-separated patterns.
    for match in _PATTERNS_DASH_SEPARATED.finditer(text):
        key = match.group("key")
        value_str = match.group("value")
        value = _safe_float(value_str)
        if value is not None and _is_valid_key(key):
            if key not in metrics:
                metrics[key] = value

    # Strategy 4: Percentage patterns (before general key-value to catch them first).
    for match in _PATTERNS_PERCENTAGE.finditer(text):
        key = match.group("key")
        value_str = match.group("value")
        value = _safe_float(value_str)
        if value is not None and _is_valid_key(key):
            metrics[key] = value

    # Strategy 5: JSON-like dict patterns (matches quoted and unquoted keys/values).
    for match in _PATTERNS_JSON_LIKE.finditer(text):
        key = match.group("key")
        value_str = match.group("value")
        value = _safe_float(value_str)
        if value is not None and _is_valid_key(key):
            if key not in metrics:
                metrics[key] = value

    # Strategy 6: General key: value / key=value patterns (lowest specificity).
    # Last match wins for duplicate keys - users expect the final value in a line.
    for match in _PATTERNS_KEY_VALUE.finditer(text):
        key = match.group("key")
        value_str = match.group("value")
        value = _safe_float(value_str)
        if value is not None and _is_valid_key(key):
            metrics[key] = value

    # Strategy 7: Special float values (nan, inf) - checked last since no digits.
    special_metrics = _parse_special_value(text)
    for key, value in special_metrics.items():
        if key not in metrics:
            metrics[key] = value

    return metrics
