"""Basic demo: simulate a training loop with print statements.

This example shows how to use PrintBoard with zero code changes to your
existing training loop. Simply add the @tb_log decorator and run TensorBoard
to visualize your metrics.

Run:
    python examples/basic_demo.py
    tensorboard --logdir=runs/basic_demo
"""

import random
import time

from printboard import tb_log


@tb_log(log_dir="runs/basic_demo")
def train():
    """Simulate a training loop with print-based metric output."""
    print("Starting training...")
    print("=" * 50)

    for epoch in range(10):
        # Simulate training metrics with some noise.
        loss = 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05)
        acc = min(0.99, 0.5 + epoch * 0.05 + random.uniform(-0.02, 0.02))
        lr = 0.001 * (0.95 ** epoch)

        # These print statements are automatically captured and parsed.
        print(f"epoch {epoch} | loss: {loss:.4f} | acc: {acc:.4f} | lr: {lr:.6f}")
        time.sleep(0.1)  # Simulate training time.

    print("=" * 50)
    print("Training completed!")


if __name__ == "__main__":
    train()
    print("\nDone! Run: tensorboard --logdir=runs/basic_demo")
