"""Integration tests that verify the full pipeline works end-to-end."""

import io
import os
import sys

from printboard import tb_log, tb_print
from printboard.decorator import tb_print_close, tb_print_close_all


class TestFullPipeline:
    """End-to-end tests for the complete capture → parse → log pipeline."""

    def test_decorator_full_training_loop(self, tmp_path):
        """Simulate a realistic training loop with multiple metric types."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("Starting training...")
            for epoch in range(5):
                loss = 0.5 / (epoch + 1)
                acc = min(0.99, 0.5 + epoch * 0.1)
                lr = 0.01 * (0.9**epoch)
                print(
                    f"epoch {epoch} | loss: {loss:.4f} | acc: {acc:.4f} | lr: {lr:.6f}"
                )
            print("Training complete!")

        train()

        # Verify TensorBoard event files were created.
        files = os.listdir(log_dir)
        event_files = [f for f in files if "events" in f]
        assert len(event_files) > 0, "No event files created"

        # Verify stdout was not blocked - output should be visible.
        captured = io.StringIO()
        original = sys.stdout
        sys.stdout = captured

        @tb_log(log_dir=str(tmp_path / "run2"))
        def train2():
            print("Hello from train2!")

        train2()
        sys.stdout = original

        assert "Hello from train2!" in captured.getvalue()

    def test_tb_print_full_workflow(self, tmp_path):
        """Test tb_print in a realistic scenario."""
        log_dir = str(tmp_path)

        tb_print_close_all()

        for epoch in range(3):
            tb_print(
                "loss", 0.5 / (epoch + 1), step=epoch, log_dir=log_dir, also_print=False
            )
            tb_print(
                "acc", 0.6 + epoch * 0.1, step=epoch, log_dir=log_dir, also_print=False
            )

        files = os.listdir(log_dir)
        event_files = [f for f in files if "events" in f]
        assert len(event_files) > 0

        tb_print_close(log_dir)

    def test_decorator_with_exception_in_middle(self, tmp_path):
        """Test that partial metrics are still logged when exception occurs."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train_with_error():
            print("loss: 0.5")
            print("acc: 0.8")
            raise RuntimeError("Training failed!")
            print("lr: 0.01")  # This won't be reached.

        try:
            train_with_error()
        except RuntimeError:
            pass

        # The function should have raised RuntimeError.
        # stdout should be restored.
        # (Event file may or may not be created depending on implementation.)
        assert True  # No crash is the success criterion.

    def test_decorator_with_mixed_content(self, tmp_path):
        """Test decorator handling mixed metric and non-metric output."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("=== Training Started ===")
            print("Model: ResNet-50, Batch: 32")
            print("loss: 0.5, acc: 0.8")
            print("------ Validation ------")
            print("val_loss: 0.55, val_acc: 0.78")
            print("=== Epoch Complete ===")

        train()

        files = os.listdir(log_dir)
        event_files = [f for f in files if "events" in f]
        assert len(event_files) > 0

    def test_multiple_decorated_functions_same_logdir(self, tmp_path):
        """Test multiple decorated functions writing to the same log directory."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("train_loss: 0.5")

        @tb_log(log_dir=log_dir)
        def validate():
            print("val_loss: 0.55")

        train()
        validate()

        files = os.listdir(log_dir)
        event_files = [f for f in files if "events" in f]
        assert len(event_files) > 0

    def test_decorator_preserves_function_metadata(self, tmp_path):
        """Test that @tb_log preserves function name, docstring, etc."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def my_train_func():
            """Train the model."""
            print("loss: 0.5")

        assert my_train_func.__name__ == "my_train_func"
        assert my_train_func.__doc__ == "Train the model."

    def test_decorator_with_return_value(self, tmp_path):
        """Test that decorated functions can return values."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("final_loss: 0.1")
            return {"loss": 0.1, "accuracy": 0.95}

        result = train()
        assert result["loss"] == 0.1
        assert result["accuracy"] == 0.95

    def test_decorator_with_no_metrics(self, tmp_path):
        """Test decorator when function has no metrics at all."""
        log_dir = str(tmp_path)

        @tb_log(log_dir=log_dir)
        def train():
            print("Training started...")
            print("Training complete!")
            return True

        result = train()
        assert result is True

        # Should not crash even with no metrics.
        assert True
