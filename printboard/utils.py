"""Utility functions for PrintBoard."""

from __future__ import annotations

import re

# Pattern to check if a string looks like a valid metric value.
_METRIC_VALUE_PATTERN = re.compile(r"^[-+]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?$")


def is_valid_metric_name(name: str, max_length: int = 32) -> bool:
    """Check if a string is a valid metric name for TensorBoard.

    Args:
        name: The metric name to validate.
        max_length: Maximum allowed length for the metric name.

    Returns:
        True if the name is valid, False otherwise.
    """
    if not name or len(name) > max_length:
        return False
    # TensorBoard accepts almost any string as a tag name,
    # but we restrict to alphanumeric, underscores, hyphens, and slashes
    # for cleaner visualization.
    return bool(re.match(r"^[a-zA-Z0-9_/\-]+$", name))


def is_metric_value(value: str) -> bool:
    """Check if a string represents a valid numeric metric value.

    Args:
        value: The string to check.

    Returns:
        True if the string is a valid numeric value.
    """
    return bool(_METRIC_VALUE_PATTERN.match(value))


def sanitize_metric_name(name: str, max_length: int = 32) -> str:
    """Sanitize a metric name for TensorBoard compatibility.

    Replaces invalid characters with underscores and truncates to max_length.

    Args:
        name: The metric name to sanitize.
        max_length: Maximum allowed length.

    Returns:
        A sanitized metric name.
    """
    # Replace spaces and invalid characters with underscores.
    sanitized = re.sub(r"[^a-zA-Z0-9_/\-]", "_", name)
    # Truncate if too long.
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized
