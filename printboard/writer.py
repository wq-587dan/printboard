"""TensorBoard writer wrapper with lifecycle management.

This module provides a thread-safe wrapper around torch.utils.tensorboard
SummaryWriter. It manages writer creation, caching by log directory,
global step tracking, and clean shutdown.

Key design decisions:
    - Writers are cached by log_dir, so multiple TBWriter instances
      pointing to the same directory share one underlying SummaryWriter.
    - Global step is shared across all instances for the same log_dir.
    - Writers are NOT closed on every call - they persist until
      explicitly closed or program exit (via atexit handler).
    - Thread-safe: uses locks for concurrent write protection.
"""

from __future__ import annotations

import atexit
import threading
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class _LogDirState:
    """Holds shared state for a single log directory.

    Ensures that all TBWriter instances pointing to the same log_dir
    share the same SummaryWriter, step counter, and lock.
    """

    __slots__ = ("writer", "lock", "step_lock", "global_step")

    def __init__(self, log_dir: str) -> None:
        """Initialize state for a log directory.

        Args:
            log_dir: Directory path for TensorBoard event files.
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.lock = threading.Lock()
        self.step_lock = threading.Lock()
        self.global_step = 0


class TBWriter:
    """Thread-safe TensorBoard writer wrapper with lifecycle management.

    Manages a single SummaryWriter instance with automatic step tracking.
    Writers are cached by log directory so multiple TBWriter instances
    pointing to the same directory share one underlying SummaryWriter.

    Attributes:
        log_dir: Directory where TensorBoard event files are written.
    """

    # Class-level cache: log_dir -> _LogDirState
    _writers: dict[str, _LogDirState] = {}
    _cache_lock = threading.Lock()

    def __init__(self, log_dir: str = "runs") -> None:
        """Initialize a TBWriter for the given log directory.

        If a writer for this log_dir already exists, the existing one is reused.

        Args:
            log_dir: Directory path for TensorBoard event files.
        """
        self.log_dir = log_dir
        self._register()

    def _register(self) -> None:
        """Register or retrieve the cached state for this log_dir.

        Uses a class-level lock to ensure only one SummaryWriter is created
        per log directory, even under concurrent access.
        """
        with self.__class__._cache_lock:
            if self.log_dir not in self.__class__._writers:
                state = _LogDirState(self.log_dir)
                self.__class__._writers[self.log_dir] = state
                # Register cleanup on exit.
                atexit.register(self.__class__._close_all)

    @property
    def writer(self) -> SummaryWriter:
        """Get the underlying SummaryWriter instance.

        Returns:
            The cached SummaryWriter for this log directory.
        """
        with self.__class__._cache_lock:
            return self.__class__._writers[self.log_dir].writer

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a scalar metric to TensorBoard.

        Args:
            tag: Metric name (e.g., "loss", "accuracy").
            value: Metric value.
            step: Optional step number. If not provided, uses the
                  shared auto-incremented global step.
        """
        if step is None:
            step = self.get_next_step()

        with self.__class__._cache_lock:
            state = self.__class__._writers[self.log_dir]
            with state.lock:
                state.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def get_next_step(self) -> int:
        """Get and increment the shared global step counter.

        The step counter is shared across all TBWriter instances
        pointing to the same log_dir.

        Returns:
            The step number before incrementing.
        """
        with self.__class__._cache_lock:
            state = self.__class__._writers[self.log_dir]
        with state.step_lock:
            step = state.global_step
            state.global_step += 1
            return step

    @property
    def global_step(self) -> int:
        """Get the current shared global step counter.

        Returns:
            The current step number.
        """
        with self.__class__._cache_lock:
            return self.__class__._writers[self.log_dir].global_step

    def reset_step(self, step: int = 0) -> None:
        """Reset the shared global step counter.

        Useful for starting a new experiment in the same log directory.

        Args:
            step: The step number to reset to. Defaults to 0.
        """
        with self.__class__._cache_lock:
            state = self.__class__._writers[self.log_dir]
        with state.step_lock:
            state.global_step = step

    def close(self) -> None:
        """Flush and close the writer for this log directory.

        This closes the shared writer for the log directory.
        After closing, a new SummaryWriter will be created on next use.
        """
        with self.__class__._cache_lock:
            if self.log_dir in self.__class__._writers:
                state = self.__class__._writers[self.log_dir]
                with state.lock:
                    state.writer.close()
                del self.__class__._writers[self.log_dir]

    @classmethod
    def _close_all(cls) -> None:
        """Close all cached writers. Called on program exit.

        This is registered as an atexit handler to ensure clean shutdown.
        """
        with cls._cache_lock:
            for state in cls._writers.values():
                with state.lock:
                    state.writer.close()
            cls._writers.clear()

    @classmethod
    def reset(cls) -> None:
        """Close and clear all cached writers.

        Primarily useful for testing to ensure clean state between tests.
        """
        cls._close_all()
