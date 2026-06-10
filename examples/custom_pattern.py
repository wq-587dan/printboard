"""Custom pattern demo: use regex for parsing specific formats.

This example shows how to use custom regex patterns with named capture groups
to parse non-standard metric output formats.

Run:
    python examples/custom_pattern.py
    tensorboard --logdir=runs/custom_pattern
"""

import random
import time

from printboard import tb_log, tb_print


# Example 1: Parse a custom format like "0.3456/0.9200"
@tb_log(log_dir="runs/custom_pattern", pattern=r"(?P<loss>[\d.]+)/(?P<acc>[\d.]+)")
def train_slash_format():
    """Simulate training with slash-separated metrics."""
    print("Training with slash format: loss/acc")
    print("-" * 40)

    for epoch in range(5):
        loss = 0.5 / (epoch + 1) + random.uniform(-0.02, 0.02)
        acc = min(0.99, 0.6 + epoch * 0.08)
        print(f"{loss:.4f}/{acc:.4f}")
        time.sleep(0.1)

    print("-" * 40)
    print("Done with slash format training!")


# Example 2: Use tb_print for explicit metric logging
def train_with_tb_print():
    """Simulate training using tb_print for explicit logging."""
    print("Training with tb_print...")
    print("-" * 40)

    for epoch in range(5):
        loss = 0.3 / (epoch + 1) + random.uniform(-0.01, 0.01)
        acc = min(0.99, 0.7 + epoch * 0.06)
        lr = 0.01 * (0.9 ** epoch)

        # tb_print explicitly logs metrics to TensorBoard.
        tb_print("tb_loss", loss, step=epoch, log_dir="runs/custom_pattern")
        tb_print("tb_acc", acc, step=epoch, log_dir="runs/custom_pattern")
        tb_print("tb_lr", lr, step=epoch, log_dir="runs/custom_pattern")

        time.sleep(0.1)

    print("-" * 40)
    print("Done with tb_print training!")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Custom regex pattern (slash-separated)")
    print("=" * 60)
    train_slash_format()

    print()
    print("=" * 60)
    print("Example 2: tb_print explicit logging")
    print("=" * 60)
    train_with_tb_print()

    print("\nDone! Run: tensorboard --logdir=runs/custom_pattern")
