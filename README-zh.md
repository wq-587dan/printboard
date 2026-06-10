# PrintBoard

[![English](https://img.shields.io/badge/lang-en-green.svg)](README.md) [![中文](https://img.shields.io/badge/lang-zh-blue.svg)](README-zh.md)

[![PyPI](https://img.shields.io/pypi/v/printboard.svg)](https://pypi.org/project/printboard/)
[![Python](https://img.shields.io/pypi/pyversions/printboard.svg)](https://pypi.org/project/printboard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/your-org/printboard/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/printboard/actions)

> 零代码修改的装饰器，将 `print()` 自动桥接到 **TensorBoard**。无需改动任何 print 代码，即可实现训练可视化。

## 为什么需要 PrintBoard？

大多数深度学习训练代码使用 `print()` 输出 loss、accuracy 等指标：

```python
print(f"epoch {epoch}, loss: {loss:.4f}, acc: {acc:.4f}")
```

但这些信息：
1. **只存在于终端**，关掉就没了
2. **无法可视化**趋势变化
3. 接入 TensorBoard 需要手动将每一行 `print()` 改为 `writer.add_scalar()`——**繁琐且容易出错**

**PrintBoard 一行装饰器解决这个问题。** 你的 print 输出自动流入 TensorBoard，终端照常显示。

## 安装

```bash
pip install printboard
```

## 快速开始

**三行代码，零修改已有的 print。**

```python
from printboard import tb_log

@tb_log(log_dir="runs/experiment_1")
def train():
    for epoch in range(100):
        loss = train_one_epoch()
        print(f"epoch {epoch}, loss: {loss:.4f}")

train()
```

然后启动 TensorBoard：

```bash
tensorboard --logdir=runs
```

就这么简单。你的 loss 曲线已经在 TensorBoard 中了。

## 支持的 Print 格式

PrintBoard 自动识别以下常见格式：

| 格式 | 示例输出 | 自动解析？ |
|------|---------|-----------|
| `key: value` | `loss: 0.3456, acc: 0.92` | 是 |
| `key=value` | `loss=0.3456, acc=0.92` | 是 |
| 管道分隔 | `epoch 3 \| loss: 0.3456 \| acc: 0.92` | 是 |
| 横线分隔 | `Step 100 - loss: 0.3456` | 是 |
| 科学计数法 | `lr: 1.5e-4` | 是 |
| 纯文本 | `训练开始...` | 忽略 |

## 自定义 Pattern

对于非标准输出格式，传入带命名捕获组的正则表达式：

```python
import re
from printboard import tb_log

@tb_log(pattern=r"(?P<loss>[\d.]+)/(?P<acc>[\d.]+)")
def train():
    print(f"{loss}/{acc}")  # 例如 "0.3456/0.9200"

train()
```

命名捕获组的名称会成为 TensorBoard 中的指标名称。

## tb_print

不通过装饰器，直接打印并记录指标：

```python
from printboard import tb_print

for epoch in range(100):
    loss = train_one_epoch()
    tb_print("loss", loss, step=epoch, log_dir="runs/my_exp")
    # 终端打印 "loss: 0.3456" 同时写入 TensorBoard
```

## API 参考

### `tb_log(log_dir="runs", pattern=None, global_step=None)`

装饰器：捕获 print 输出并将指标写入 TensorBoard。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `log_dir` | `str` | `"runs"` | TensorBoard 日志目录 |
| `pattern` | `str \| re.Pattern \| None` | `None` | 自定义正则表达式（命名捕获组） |
| `global_step` | `int \| None` | `None` | 起始步数 |

### `tb_print(tag, value, step=None, log_dir="runs", also_print=True)`

直接打印并记录指标到 TensorBoard。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tag` | `str` | -- | 指标名称（如 "loss"） |
| `value` | `float` | -- | 指标值 |
| `step` | `int \| None` | `None` | 步数（不指定则自动递增） |
| `log_dir` | `str` | `"runs"` | TensorBoard 日志目录 |
| `also_print` | `bool` | `True` | 是否同时在终端打印 |

## 运行 TensorBoard

```bash
# 基本用法
tensorboard --logdir=runs

# 指定端口
tensorboard --logdir=runs --port=6006

# 在浏览器中打开
tensorboard --logdir=runs --host=localhost --port=6006
```

然后在浏览器中打开 `http://localhost:6006`。

## 项目结构

```
printboard/
├── printboard/
│   ├── __init__.py        # 包导出
│   ├── decorator.py       # tb_log 装饰器 + tb_print
│   ├── parser.py          # print 输出解析器
│   ├── writer.py          # TensorBoard 写入封装
│   └── utils.py           # 工具函数
├── examples/
│   ├── basic_demo.py      # 基础用法演示
│   └── custom_pattern.py  # 自定义 pattern 演示
├── tests/
│   ├── test_parser.py     # 解析器单元测试
│   └── test_decorator.py  # 装饰器单元测试
├── README.md
├── README-zh.md
├── LICENSE
├── pyproject.toml
└── setup.py
```

## 贡献指南

我们欢迎贡献！以下是开发流程：

1. **Fork** 本仓库
2. **从 main 创建分支**：`git checkout -b feat/your-feature`
3. **安装开发依赖**：`pip install -e ".[dev]"`
4. **进行修改并编写测试**
5. **运行测试**：`pytest tests/ -v --cov=printboard`
6. **提交代码**，遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：
   ```
   feat(parser): 支持 JSON 格式指标
   fix(decorator): 异常时恢复 stdout
   ```
7. **推送并创建 Pull Request** 到 `main`

### 开发规范

- 所有公开函数使用类型注解
- Google 风格 docstring
- 测试覆盖率 >= 80%
- 行宽：88 字符（Black 默认）
- 错误信息使用英文

## License

MIT License. 详见 [LICENSE](LICENSE)。

## 致谢

基于 PyTorch TensorBoard 集成构建。灵感来源于手动将 print 转换为日志记录的痛苦经历。
