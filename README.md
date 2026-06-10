# PrintBoard

[![English](https://img.shields.io/badge/lang-en-green.svg)](README.md) [![中文](https://img.shields.io/badge/lang-zh-blue.svg)](README-zh.md)

[![PyPI](https://img.shields.io/pypi/v/printboard.svg)](https://pypi.org/project/printboard/)
[![Python](https://img.shields.io/pypi/pyversions/printboard.svg)](https://pypi.org/project/printboard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/your-org/printboard/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/printboard/actions)

> Zero-code decorator to bridge `print()` to **TensorBoard**. Visualize your training metrics without modifying a single line of print code.

## Problem

Most deep learning training scripts use `print()` to output loss, accuracy, and other metrics:

```python
print(f"epoch {epoch}, loss: {loss:.4f}, acc: {acc:.4f}")
```

This information is **lost when the terminal closes**, and integrating with TensorBoard requires replacing every `print()` with `writer.add_scalar()` -- a tedious, error-prone process.

**PrintBoard eliminates this friction.** Add one decorator, and your metrics flow automatically into TensorBoard.

## Installation

```bash
pip install printboard
```

## Quick Start

**Three lines. Zero changes to your print statements.**

```python
from printboard import tb_log

@tb_log(log_dir="runs/experiment_1")
def train():
    for epoch in range(100):
        loss = train_one_epoch()
        print(f"epoch {epoch}, loss: {loss:.4f}")

train()
```

Then launch TensorBoard:

```bash
tensorboard --logdir=runs
```

That's it. Your loss curve is now visible in TensorBoard.

## Supported Print Formats

PrintBoard automatically recognizes these common formats:

| Format | Example Output | Auto-parsed? |
|--------|---------------|--------------|
| `key: value` | `loss: 0.3456, acc: 0.92` | Yes |
| `key=value` | `loss=0.3456, acc=0.92` | Yes |
| Pipe-separated | `epoch 3 \| loss: 0.3456 \| acc: 0.92` | Yes |
| Dash-separated | `Step 100 - loss: 0.3456` | Yes |
| Scientific notation | `lr: 1.5e-4` | Yes |
| Pure text | `Training started...` | Ignored |

## Custom Pattern

For non-standard output formats, pass a regex with named capture groups:

```python
import re
from printboard import tb_log

@tb_log(pattern=r"(?P<loss>[\d.]+)/(?P<acc>[\d.]+)")
def train():
    print(f"{loss}/{acc}")  # e.g., "0.3456/0.9200"

train()
```

Named group names become TensorBoard metric tags.

## tb_print

For explicit metric logging without the decorator:

```python
from printboard import tb_print

for epoch in range(100):
    loss = train_one_epoch()
    tb_print("loss", loss, step=epoch, log_dir="runs/my_exp")
    # Prints: "loss: 0.3456" to terminal AND logs to TensorBoard
```

## API Reference

### `tb_log(log_dir="runs", pattern=None, global_step=None)`

Decorator that captures print output and logs metrics to TensorBoard.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | `str` | `"runs"` | Directory for TensorBoard event files |
| `pattern` | `str \| re.Pattern \| None` | `None` | Custom regex with named capture groups |
| `global_step` | `int \| None` | `None` | Starting step number |

### `tb_print(tag, value, step=None, log_dir="runs", also_print=True)`

Directly print and log a metric to TensorBoard.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tag` | `str` | -- | Metric name (e.g., "loss") |
| `value` | `float` | -- | Metric value |
| `step` | `int \| None` | `None` | Step number (auto-increments if not provided) |
| `log_dir` | `str` | `"runs"` | Directory for TensorBoard event files |
| `also_print` | `bool` | `True` | Whether to also print to stdout |

## Running TensorBoard

```bash
# Basic usage
tensorboard --logdir=runs

# Specify a port
tensorboard --logdir=runs --port=6006

# Open in browser
tensorboard --logdir=runs --host=localhost --port=6006
```

Then open `http://localhost:6006` in your browser.

## Project Structure

```
printboard/
├── printboard/
│   ├── __init__.py        # Package exports
│   ├── decorator.py       # tb_log decorator + tb_print
│   ├── parser.py          # Print output parser
│   ├── writer.py          # TensorBoard writer wrapper
│   └── utils.py           # Utility functions
├── examples/
│   ├── basic_demo.py      # Basic usage demo
│   └── custom_pattern.py  # Custom pattern demo
├── tests/
│   ├── test_parser.py     # Parser unit tests
│   └── test_decorator.py  # Decorator unit tests
├── README.md
├── README-zh.md
├── LICENSE
├── pyproject.toml
└── setup.py
```

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feat/your-feature`
3. **Install dev dependencies**: `pip install -e ".[dev]"`
4. **Make your changes** with tests
5. **Run tests**: `pytest tests/ -v --cov=printboard`
6. **Commit** following [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(parser): add support for JSON format metrics
   fix(decorator): restore stdout on exception
   ```
7. **Push** and open a Pull Request to `main`

### Development Guidelines

- Type hints on all public functions
- Google-style docstrings
- Test coverage >= 80%
- Line length: 88 characters (Black)
- Error messages in English

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built with PyTorch TensorBoard integration. Inspired by the frustration of manually converting print statements to logging calls.
