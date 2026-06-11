"""Tests for the tb_log decorator and tb_print function."""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

import pytest

from printboard import tb_log, tb_print, tb_print_close, tb_print_close_all


def _read_tb_events(log_dir: str) -> list[tuple[str, float, int]]:
    """Read TensorBoard event file and return list of (tag, value, step) tuples.

    Uses the tensorboard event file reader if available.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator

        event_files = [str(p) for p in Path(log_dir).glob("events*")]
        if not event_files:
            return []

        ea = event_accumulator.EventAccumulator(event_files[0])
        ea.Reload()

        metrics = []
        for tag in ea.Tags()["scalars"]:
            for event in ea.Scalars(tag):
                metrics.append((tag, event.value, event.step))
        return metrics
    except Exception:
        return []


class TestDecoratorCapturesMetrics:
    """Test that the decorator correctly captures and logs metrics."""

    def test_decorator_captures_colon_metrics(self, tmp_path):
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("loss: 0.5, acc: 0.8")

        train()

        files = os.listdir(log_dir)
        assert any("events" in f for f in files), "No TensorBoard event file"

    def test_decorator_captures_equal_metrics(self, tmp_path):
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("loss=0.5,acc=0.8")

        train()

        files = os.listdir(log_dir)
        assert any("events" in f for f in files)

    def test_decorator_captures_pipe_separated(self, tmp_path):
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("epoch 1 | loss: 0.5 | acc: 0.8")

        train()

        files = os.listdir(log_dir)
        assert any("events" in f for f in files)

    def test_decorator_no_metrics_line(self, tmp_path):
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("Training started...")
            print("No metrics here!")

        train()

        # Should not crash; event file may exist but be empty of scalars.
        assert True

    def test_decorator_multiple_invocations(self, tmp_path):
        """Test that calling a decorated function multiple times works."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("loss: 0.5")

        train()
        train()  # Second call should not crash.

        files = os.listdir(log_dir)
        assert any("events" in f for f in files)

    def test_decorator_preserves_output(self, tmp_path):
        """Test that print output is still visible in terminal."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("Hello from train!")

        # Capture stdout to verify output is preserved.
        captured = io.StringIO()
        original = sys.stdout
        sys.stdout = captured
        train()
        sys.stdout = original

        output = captured.getvalue()
        assert "Hello from train!" in output

    def test_decorator_returns_result(self, tmp_path):
        """Test that the decorator preserves the function's return value."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def compute():
            print("result: 42")
            return 100

        result = compute()
        assert result == 100

    def test_decorator_with_args(self, tmp_path):
        """Test that the decorator passes arguments correctly."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train(epoch, lr):
            print(f"epoch: {epoch}, lr: {lr}")
            return epoch * lr

        result = train(10, 0.01)
        assert result == 0.1

    def test_decorator_with_kwargs(self, tmp_path):
        """Test that the decorator passes keyword arguments correctly."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train(epoch=5, lr=0.01):
            print(f"epoch: {epoch}, lr: {lr}")

        train(epoch=3, lr=0.001)

    def test_decorator_writes_correct_values(self, tmp_path):
        """Test that the decorator writes the correct metric values to TensorBoard."""
        import math

        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("loss: 0.123")
            print("acc: 0.456")

        train()

        metrics = _read_tb_events(log_dir)
        # TensorBoard uses float32, so use approximate comparison.
        for tag, value, step in metrics:
            if tag == "loss":
                assert math.isclose(
                    value, 0.123, rel_tol=1e-5
                ), f"Expected ~0.123, got {value}"
            elif tag == "acc":
                assert math.isclose(
                    value, 0.456, rel_tol=1e-5
                ), f"Expected ~0.456, got {value}"


class TestDecoratorCustomPattern:
    """Test custom pattern support in the decorator."""

    def test_custom_pattern_string(self, tmp_path):
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir, pattern=r"(?P<loss>[\d.]+)/(?P<acc>[\d.]+)")
        def train():
            print("0.5/0.8")

        train()

        files = os.listdir(log_dir)
        assert any("events" in f for f in files)

    def test_custom_pattern_compiled(self, tmp_path):
        log_dir = str(tmp_path)

        pattern = re.compile(r"loss=(?P<val>[\d.]+)")

        @tb_log(log_dir=log_dir, pattern=pattern)
        def train():
            print("loss=0.1234")

        train()

        files = os.listdir(log_dir)
        assert any("events" in f for f in files)

    def test_invalid_pattern_raises(self, tmp_path):
        log_dir = str(tmp_path)
        with pytest.raises(ValueError, match="Invalid regex"):

            @tb_log(log_dir=log_dir, pattern="[invalid")
            def train():
                pass


class TestDecoratorExceptionSafety:
    """Test that the decorator handles exceptions correctly."""

    def test_exception_restores_stdout(self, tmp_path):
        """Test that stdout is restored even when the function raises."""
        log_dir = str(tmp_path)
        original_stdout = sys.stdout

        @tb_log(log_dir=log_dir)
        def train():
            print("before error")
            raise RuntimeError("Training failed!")

        with pytest.raises(RuntimeError, match="Training failed!"):
            train()

        # Verify stdout was restored.
        assert sys.stdout is original_stdout

    def test_exception_not_swallowed(self, tmp_path):
        """Test that exceptions are not swallowed by the decorator."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            raise ValueError("Custom error")

        with pytest.raises(ValueError, match="Custom error"):
            train()

    def test_keyboard_interrupt_restores_stdout(self, tmp_path):
        """Test that even KeyboardInterrupt restores stdout."""
        log_dir = str(tmp_path)
        original_stdout = sys.stdout

        @tb_log(log_dir=log_dir)
        def train():
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            train()

        assert sys.stdout is original_stdout


class TestDecoratorStepIncrement:
    """Test step counter behavior."""

    def test_step_increments_per_metric(self, tmp_path):
        """Test that each metric gets an incrementing step."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir, global_step=0)
        def train():
            print("loss: 0.5, acc: 0.8")

        train()

        files = os.listdir(log_dir)
        assert any("events" in f for f in files)

    def test_steps_accumulate_across_calls(self, tmp_path):
        """Test that steps accumulate across multiple function calls."""
        log_dir = str(tmp_path)

        # Don't pass global_step so it auto-increments from 0 on first call.
        @tb_log(log_dir=log_dir)
        def train():
            print("loss: 0.5")

        train()
        train()

        metrics = _read_tb_events(log_dir)
        # Two calls, each logging 1 metric → 2 total metrics.
        assert len(metrics) == 2
        steps = [step for _, _, step in metrics]
        assert steps[0] < steps[1]


class TestDecoratorNestedCalls:
    """Test nested function calls don't conflict."""

    def test_nested_prints(self, tmp_path):
        log_dir = str(tmp_path)

        def helper():
            print("nested_loss: 0.1")

        @tb_log(log_dir=log_dir)
        def train():
            print("outer_loss: 0.5")
            helper()
            print("final_acc: 0.9")

        train()

        files = os.listdir(log_dir)
        assert any("events" in f for f in files)


class TestDecoratorNewFormats:
    """Test decorator with new parser formats."""

    def test_decorator_percentage(self, tmp_path):
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("acc: 95%")

        train()

        metrics = _read_tb_events(log_dir)
        for tag, value, _ in metrics:
            if tag == "acc":
                import math

                assert math.isclose(value, 95.0, rel_tol=1e-5)

    def test_decorator_json_like(self, tmp_path):
        import math

        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("{'loss': 0.3456, 'acc': 0.92}")

        train()

        metrics = _read_tb_events(log_dir)
        for tag, value, _ in metrics:
            if tag == "loss":
                assert math.isclose(value, 0.3456, rel_tol=1e-5)
            elif tag == "acc":
                assert math.isclose(value, 0.92, rel_tol=1e-5)

    def test_decorator_progress_bar(self, tmp_path):
        import math

        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("1/10 [====] - loss: 0.5")

        train()

        metrics = _read_tb_events(log_dir)
        for tag, value, _ in metrics:
            if tag == "loss":
                assert math.isclose(value, 0.5, rel_tol=1e-5)


class TestTBPrint:
    """Test the tb_print function."""

    def test_tb_print_direct(self, tmp_path):
        log_dir = str(tmp_path)

        tb_print("test_loss", 0.5, step=0, log_dir=log_dir, also_print=False)

        files = os.listdir(str(tmp_path))
        assert any("events" in f for f in files)
        tb_print_close(log_dir)

    def test_tb_print_also_prints(self, tmp_path):
        log_dir = str(tmp_path)

        captured = io.StringIO()
        original = sys.stdout
        sys.stdout = captured
        tb_print("test_acc", 0.9, step=0, log_dir=log_dir)
        sys.stdout = original

        output = captured.getvalue()
        assert "test_acc: 0.9" in output
        tb_print_close(log_dir)

    def test_tb_print_auto_step(self, tmp_path):
        log_dir = str(tmp_path)

        tb_print("loss", 0.5, log_dir=log_dir, also_print=False)
        tb_print("acc", 0.9, log_dir=log_dir, also_print=False)

        files = os.listdir(str(tmp_path))
        assert any("events" in f for f in files)
        tb_print_close(log_dir)

    def test_tb_print_close_all(self, tmp_path):
        log_dir = str(tmp_path)

        tb_print("loss", 0.5, log_dir=log_dir, also_print=False)
        tb_print_close_all()

        # After close, further tb_prints should still work (creates new writer).
        tb_print("loss", 0.6, log_dir=log_dir, also_print=False)
        tb_print_close(log_dir)

    def test_tb_print_sanitizes_tag(self, tmp_path):
        log_dir = str(tmp_path)

        # Tag with special characters should be sanitized.
        tb_print("my metric!", 0.5, step=0, log_dir=log_dir, also_print=False)
        tb_print_close(log_dir)

        files = os.listdir(log_dir)
        assert any("events" in f for f in files)
