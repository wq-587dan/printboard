"""Core decorator for capturing print output and logging to TensorBoard.

This module provides the tb_log decorator and tb_print function, which are
the main entry points for PrintBoard users. The decorator intercepts
sys.stdout during function execution, captures print output, parses metrics,
and writes them to TensorBoard — all without modifying the original code.
"""

from __future__ import annotations

import io
import re
import sys
import threading
from functools import wraps
from typing import Callable, Optional, Union

from printboard.parser import parse_print_output
from printboard.utils import sanitize_metric_name
from printboard.writer import TBWriter


class StreamProxy:
    """A file-like object that writes to both original stdout and an internal buffer.

    This class enables dual-write: every write to sys.stdout goes to both
    the original stdout (so the user still sees print output in the terminal)
    and an internal StringIO buffer (for later parsing).

    Thread-safe: uses a lock to prevent interleaved writes.
    """

    def __init__(self, original: io.TextIOWrapper, buffer: io.StringIO) -> None:
        """Initialize the stream proxy.

        Args:
            original: The original sys.stdout to forward writes to.
            buffer: The internal buffer to capture writes into.
        """
        self._original = original
        self._buffer = buffer
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        """Write text to both the original stdout and the internal buffer.

        Args:
            text: The text to write.

        Returns:
            The number of characters written.
        """
        with self._lock:
            self._buffer.write(text)
            try:
                self._original.write(text)
                self._original.flush()
            except OSError:
                # Terminal may be closed or redirected; ignore write errors
                # to the original stream while preserving the buffer.
                pass
            return len(text)

    def flush(self) -> None:
        """Flush both the original stdout and the internal buffer."""
        with self._lock:
            try:
                self._original.flush()
            except OSError:
                pass
            self._buffer.flush()

    def fileno(self) -> int:
        """Return the file descriptor of the original stdout.

        Required for compatibility with code that checks sys.stdout.fileno().
        """
        return self._original.fileno()

    def isatty(self) -> bool:
        """Return whether the original stdout is a TTY."""
        return self._original.isatty()


def tb_log(
    log_dir: str = "runs",
    pattern: Optional[Union[str, re.Pattern[str]]] = None,
    global_step: Optional[int] = None,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Decorator that captures print output and logs metrics to TensorBoard.

    This decorator replaces sys.stdout with a StreamProxy during the wrapped
    function's execution. After the function completes, it parses all captured
    output for metric key-value pairs and writes them to TensorBoard.

    The original function's print output is still displayed in the terminal.
    Exceptions raised by the wrapped function are propagated normally, and
    stdout is always restored in the finally block.

    Args:
        log_dir: Directory for TensorBoard event files.
        pattern: Optional custom regex pattern with named capture groups.
                 Group names become metric tags, group values become metric values.
                 Can be a string or compiled Pattern.
        global_step: Optional starting step number. If not provided,
                     the writer's auto-increment is used.

    Returns:
        A decorator function that wraps the target function.

    Example:
        >>> @tb_log(log_dir="runs/my_experiment")
        ... def train():
        ...     for i in range(10):
        ...         print(f"epoch {i}, loss: {0.5 / (i + 1):.4f}")
        ...     print("Training complete!")
        ...
        >>> train()

    Raises:
        ValueError: If the custom pattern string is not a valid regex.
    """
    # Compile pattern string to Pattern object if needed.
    compiled_pattern: Optional[re.Pattern[str]] = None
    if isinstance(pattern, str):
        try:
            compiled_pattern = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern}") from e
    elif pattern is not None:
        compiled_pattern = pattern

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Save original stdout for restoration.
            original_stdout = sys.stdout
            buffer = io.StringIO()

            # Replace stdout with a dual-write proxy.
            sys.stdout = StreamProxy(original_stdout, buffer)

            try:
                # Execute the original function.
                result = func(*args, **kwargs)
            except BaseException:
                # Re-raise the exception without swallowing it.
                raise
            finally:
                # Always restore original stdout.
                sys.stdout = original_stdout
                # Ensure buffer is flushed before reading.
                buffer.flush()

            # Parse captured output and log metrics to TensorBoard.
            _capture_and_log(buffer, log_dir, compiled_pattern, global_step)

            return result

        return wrapper

    return decorator


def _capture_and_log(
    buffer: io.StringIO,
    log_dir: str,
    custom_pattern: Optional[re.Pattern[str]],
    global_step: Optional[int],
) -> None:
    """Parse captured output from buffer and log metrics to TensorBoard.

    The writer is cached by log_dir and NOT closed after each call,
    so metrics accumulate across multiple function invocations.

    Args:
        buffer: StringIO buffer containing captured stdout output.
        log_dir: Directory for TensorBoard event files.
        custom_pattern: Optional custom regex pattern for parsing.
        global_step: Optional starting step number.
    """
    writer = TBWriter(log_dir=log_dir)

    if global_step is not None:
        writer.reset_step(global_step)

    captured = buffer.getvalue()
    lines = captured.splitlines()

    for line in lines:
        if not line.strip():
            continue

        metrics = parse_print_output(line, custom_pattern)
        if not metrics:
            continue

        for tag, value in metrics.items():
            step = writer.get_next_step()
            writer.log_scalar(tag=tag, value=value, step=step)

    # Note: We do NOT close the writer here. It remains cached for
    # subsequent calls. The atexit handler will close it on program exit,
    # or the user can call tb_print_close() explicitly.


# Global writer instance for tb_print function.
# This allows tb_print to work without a decorator context.
_tb_print_writers: dict[str, TBWriter] = {}
_tb_print_lock = threading.Lock()


def tb_print(
    tag: str,
    value: float,
    step: Optional[int] = None,
    log_dir: str = "runs",
    also_print: bool = True,
) -> None:
    """Directly print a metric and log it to TensorBoard.

    This function is equivalent to calling print() but also writes the
    metric to TensorBoard. It can be used without the tb_log decorator
    for explicit metric logging.

    Args:
        tag: Metric name (e.g., "loss", "accuracy"). Invalid characters
             are automatically replaced with underscores.
        value: Metric value.
        step: Optional step number. If not provided, auto-increments.
        log_dir: Directory for TensorBoard event files.
        also_print: If True, also print the metric to stdout.

    Example:
        >>> tb_print("learning_rate", 0.001, step=0)
        learning_rate: 0.001
    """
    # Sanitize tag for TensorBoard compatibility.
    safe_tag = sanitize_metric_name(tag)

    if also_print:
        print(f"{safe_tag}: {value}")

    with _tb_print_lock:
        if log_dir not in _tb_print_writers:
            _tb_print_writers[log_dir] = TBWriter(log_dir=log_dir)

    writer = _tb_print_writers[log_dir]
    writer.log_scalar(tag=safe_tag, value=value, step=step)


def tb_print_close(log_dir: str = "runs") -> None:
    """Close the tb_print writer for the given log directory.

    Call this at the end of your script to ensure all metrics are flushed
    to disk. If not called, the atexit handler will close writers automatically.

    Args:
        log_dir: Directory whose writer should be closed.
    """
    with _tb_print_lock:
        if log_dir in _tb_print_writers:
            _tb_print_writers[log_dir].close()
            del _tb_print_writers[log_dir]


def tb_print_close_all() -> None:
    """Close all tb_print writers.

    Useful for cleanup at the end of a script or test.
    """
    with _tb_print_lock:
        for writer in _tb_print_writers.values():
            writer.close()
        _tb_print_writers.clear()
