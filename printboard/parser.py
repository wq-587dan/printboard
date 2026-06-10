"""Print output parser for extracting metrics from text.

This module provides regex-based parsing of print statements to extract
numeric metric key-value pairs. It supports multiple common formats
used in deep learning training loops and allows custom regex patterns.
"""

from __future__ import annotations

import re
from typing import Optional


# Maximum allowed key length to avoid capturing meaningless long strings.
_MAX_KEY_LENGTH = 32

# Precompiled pattern for key: value format (colon separator).
# Captures alphanumeric+underscore keys and numeric values (including scientific notation).
_PATTERNS_KEY_VALUE = re.compile(
    r"(?P<key>[\w]{1,32})\s*[:=]\s*(?P<value>-?[\d]+(?:\.[\d]*)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Precompiled pattern for pipe-separated sections containing key: value pairs.
# Handles formats like: "epoch 3 | loss: 0.3456 | acc: 0.92"
_PATTERNS_PIPE_SEPARATED = re.compile(
    r"\|\s*(?P<key>[\w]{1,32})\s*:\s*(?P<value>-?[\d]+(?:\.[\d]*)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Precompiled pattern for dash-separated sections containing key: value pairs.
# Handles formats like: "Step 100 - loss: 0.3456"
_PATTERNS_DASH_SEPARATED = re.compile(
    r"-+\s*(?P<key>[\w]{1,32})\s*:\s*(?P<value>-?[\d]+(?:\.[\d]*)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Pattern to quickly check if a line contains any numeric value worth parsing.
# Used as an early-exit optimization to skip pure text lines.
_HAS_NUMBER = re.compile(r"[\d]+(?:\.[\d]*)?(?:[eE][+-]?\d+)?")


def _is_valid_key(key: str) -> bool:
    """Check if a key is valid for TensorBoard logging.

    Filters out common non-metric words that happen to match the pattern
    but are not actual metrics (e.g., 'epoch' in 'epoch 3' context,
    'step', 'iteration').

    Args:
        key: The key string to validate.

    Returns:
        True if the key is a valid metric name.
    """
    # Allow all keys by default; users can filter if needed.
    # The key must be non-empty and within length limits.
    return 0 < len(key) <= _MAX_KEY_LENGTH


def _safe_float(value_str: str) -> Optional[float]:
    """Safely convert a string to float, returning None on failure.

    Args:
        value_str: The string to convert.

    Returns:
        The float value, or None if conversion fails.
    """
    try:
        return float(value_str)
    except (ValueError, OverflowError):
        return None


def parse_print_output(
    text: str,
    custom_pattern: Optional[re.Pattern[str]] = None,
) -> dict[str, float]:
    """Parse a single line of print output into metric key-value pairs.

    This function attempts to extract numeric metrics from text using
    multiple regex strategies. It supports common formats like "loss: 0.5",
    "acc=0.92", pipe-separated, and dash-separated formats.

    The parsing follows a priority order:
    1. Custom pattern (if provided) - highest priority
    2. Pipe-separated patterns
    3. Dash-separated patterns
    4. General key: value / key=value patterns

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

    # Early exit: skip lines with no numeric content.
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
            # Only add if not already captured by pipe strategy.
            if key not in metrics:
                metrics[key] = value

    # Strategy 4: General key: value / key=value patterns (lowest specificity).
    # Last match wins for duplicate keys — users expect the final value in a line.
    for match in _PATTERNS_KEY_VALUE.finditer(text):
        key = match.group("key")
        value_str = match.group("value")
        value = _safe_float(value_str)
        if value is not None and _is_valid_key(key):
            metrics[key] = value

    return metrics
