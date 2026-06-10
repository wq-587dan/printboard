"""TensorBoard writer wrapper with lifecycle management.

This module provides a thread-safe wrapper around torch.utils.tensorboard
SummaryWriter. It manages writer creation, caching by log directory,
global step tracking, and clean shutdown.
"""

from __future__ import annotations

import atexit
import os
import threading
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class TBWriter:
    """Thread-safe TensorBoard writer wrapper with lifecycle management.

    Manages a single SummaryWriter instance with automatic step tracking.
    Writers are cached by log directory so multiple TBWriter instances
    pointing to the same directory share one underlying SummaryWriter.

    Attributes:
        log_dir: Directory where TensorBoard event files are written.
        global_step: Current global step counter.
    """

    # Class-level cache: log_dir -> (writer_instance, lock)
    # Shared across instances to avoid multiple SummaryWriters on same dir.
    _writers: dict[str, tuple[SummaryWriter, threading.Lock]] = {}
    _cache_lock = threading.Lock()

    def __init__(self, log_dir: str = "runs") -> None:
        """Initialize a TBWriter for the given log directory.

        Args:
            log_dir: Directory path for TensorBoard event files.
        """
        self.log_dir = log_dir
        self._global_step: int = 0
        self._step_lock = threading.Lock()
        # Register this instance's lock for thread-safe access.
        self._register()

    def _register(self) -> None:
        """Register or retrieve the cached SummaryWriter for this log_dir.

        Uses a class-level lock to ensure only one SummaryWriter is created
        per log directory, even under concurrent access.
        """
        with self.__class__._cache_lock:
            if self.log_dir not in self.__class__._writers:
                writer = SummaryWriter(log_dir=self.log_dir)
                lock = threading.Lock()
                self.__class__._writers[self.log_dir] = (writer, lock)
                # Register cleanup on exit.
                atexit.register(self.__class__._close_all)

    @property
    def writer(self) -> SummaryWriter:
        """Get the underlying SummaryWriter instance.

        Returns:
            The cached SummaryWriter for this log directory.
        """
        with self.__class__._cache_lock:
            return self.__class__._writers[self.log_dir][0]

    @property
    def _writer_lock(self) -> threading.Lock:
        """Get the lock for the underlying writer."""
        with self.__class__._cache_lock:
            return self.__class__._writers[self.log_dir][1]

    @property
    def global_step(self) -> int:
        """Get the current global step counter.

        Returns:
            The current step number.
        """
        return self._global_step

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
                  auto-incremented global step.
        """
        if step is None:
            step = self.get_next_step()

        with self._writer_lock:
            self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def get_next_step(self) -> int:
        """Get and increment the global step counter.

        Returns:
            The step number before incrementing.
        """
        with self._step_lock:
            step = self._global_step
            self._global_step += 1
            return step

    def close(self) -> None:
        """Flush and close the writer for this log directory.

        Note: This closes the shared writer for the log directory.
        Subsequent writes to the same directory will create a new writer.
        """
        with self.__class__._cache_lock:
            if self.log_dir in self.__class__._writers:
                writer, lock = self.__class__._writers[self.log_dir]
                with lock:
                    writer.close()
                del self.__class__._writers[self.log_dir]

    @classmethod
    def _close_all(cls) -> None:
        """Close all cached writers. Called on program exit.

        This is registered as an atexit handler to ensure clean shutdown.
        """
        with cls._cache_lock:
            for log_dir in list(cls._writers.keys()):
                writer, lock = cls._writers[log_dir]
                with lock:
                    writer.close()
            cls._writers.clear()

    @classmethod
    def reset(cls) -> None:
        """Close and clear all cached writers.

        Primarily useful for testing to ensure clean state between tests.
        """
        cls._close_all()
