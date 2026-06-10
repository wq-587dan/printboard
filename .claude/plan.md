# printboard 项目实施方案

## 一 · 项目定位与架构总览

**一句话**：零代码修改的 print → TensorBoard 自动桥接器。

```
┌─────────────────────────────────────────────────────────────┐
│                        用户代码                              │
│                   print(f"loss: {loss}")                    │
└──────────────────────────┬──────────────────────────────────┘
                           │ sys.stdout.write()
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   StreamProxy (decorator.py)                 │
│  ┌─────────────────────┐    ┌──────────────────────────┐    │
│  │  双写到原始 stdout   │    │  缓冲到内部 buffer       │    │
│  └─────────────────────┘    └───────────┬──────────────┘    │
└─────────────────────────────────────────┼───────────────────┘
                                          │ 逐行解析
                                          ▼
┌─────────────────────────────────────────────────────────────┐
│                Parser (parser.py)                           │
│  多模式正则匹配 → extract metrics: {tag: value}             │
└──────────────────────────┬──────────────────────────────────┘
                           │ 写入指标
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Writer (writer.py)                             │
│  SummaryWriter.log_scalar(tag, value, step) → disk         │
└─────────────────────────────────────────────────────────────┘
```

---

## 二 · 开发规范（规则设定）

### 2.1 代码风格

| 维度 | 规范 |
|------|------|
| Type hints | 所有函数签名必须带类型注解，返回类型不可省略 |
| Docstring | Google 风格，每个 public 函数/类必须包含 `Args`/`Returns`/`Raises` |
| 命名 | 模块/函数/变量用 snake_case，类用 PascalCase，常量用 UPPER_SNAKE |
| 注释 | 解释"为什么"而非"是什么"；复杂正则必须注释其匹配意图 |
| 行宽 | 88 字符（Black 默认） |
| 导入顺序 | 标准库 → 第三方 → 本地，各组之间空一行 |
| 错误信息 | 英文（面向国际化） |

### 2.2 Git 规范

```
# 提交信息格式（Conventional Commits）
<type>(<scope>): <description>

type: feat | fix | docs | style | refactor | test | chore | ci
scope: parser | writer | decorator | utils | readme | setup

# 示例
feat(parser): add support for pipe-separated metrics format
fix(decorator): restore stdout on exception in wrapped function
```

### 2.3 分支与 PR 流程

```
main (受保护，不可直接 push)
  └── feat/xxx / fix/xxx / chore/xxx (功能分支)
        └── Pull Request → main
```

**PR 规则**：
- 每个 PR 对应一个功能或修复，保持原子性
- PR 描述包含：改动内容、测试覆盖、相关 issue
- 至少自审一遍代码后再提 PR
- CI 全部通过方可合并

### 2.4 测试规范

- 测试覆盖率目标：核心模块 ≥ 90%
- 测试命名：`test_<功能>_<场景>_<预期>`
- 使用 pytest 标准约定，无需 mock 外部库的内部实现
- 边界测试包含：空输入、异常输入、极端值、多线程场景

---

## 三 · 分步实施方案

### 第一步：项目骨架初始化

**目标**：建立可工作的项目结构、git 仓库、基础配置

**文件清单**：

```
printboard/
├── .gitignore
├── LICENSE
├── requirements.txt
├── setup.py
├── pyproject.toml
├── printboard/
│   ├── __init__.py
│   ├── decorator.py
│   ├── parser.py
│   ├── writer.py
│   └── utils.py
├── examples/
│   ├── basic_demo.py
│   └── custom_pattern.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_parser.py
│   └── test_decorator.py
└── .github/
    └── workflows/
        └── ci.yml
```

**细节**：

1. **`.gitignore`**：Python 标准模板（__pycache__、*.egg-info、.venv、.pytest_cache、.mypy_cache、*.pyc、.DS_Store、runs/、logs/）

2. **`LICENSE`**：MIT License，版权持有人留占位

3. **`requirements.txt`**：
   ```
   torch>=1.8.0
   tensorboard>=2.8.0
   ```

4. **`pyproject.toml`**：现代 Python 项目配置
   ```toml
   [build-system]
   requires = ["setuptools>=61.0", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "printboard"
   version = "0.1.0"
   description = "Zero-code decorator to bridge print() to TensorBoard"
   readme = "README.md"
   license = {text = "MIT"}
   requires-python = ">=3.8"
   dependencies = ["torch>=1.8.0", "tensorboard>=2.8.0"]

   [project.optional-dependencies]
   dev = ["pytest>=7.0", "pytest-cov", "black", "flake8"]

   [tool.pytest.ini_options]
   testpaths = ["tests"]

   [tool.black]
   line-length = 88
   ```

5. **`setup.py`**：最小化，指向 pyproject.toml
   ```python
   from setuptools import setup
   setup()  # 配置全在 pyproject.toml
   ```

6. **初始化 git**：`git init`，配置不受信任的 origin（后续用户替换）

---

### 第二步：实现 parser.py（核心解析引擎）

**设计要点**：

1. **多策略解析器模式**：定义一个 `PatternStrategy`，每个策略负责一种格式，按优先级顺序尝试匹配

2. **预编译正则**：所有 `re.compile()` 在模块级完成，避免运行时重复编译

3. **核心函数签名**：
   ```python
   def parse_print_output(text: str, custom_pattern: Optional[re.Pattern] = None) -> dict[str, float]:
       """Parse a single line of print output into metric key-value pairs."""
   ```

4. **支持的内置模式**（按优先级排列）：

   | 优先级 | 模式 | 示例 | 正则 |
   |--------|------|------|------|
   | 1 | `key: value` | `loss: 0.3456` | `(?P<key>[\w]+)\s*:\s*(?P<value>[\d.eE+-]+)` |
   | 2 | `key=value` | `loss=0.3456` | `(?P<key>[\w]+)\s*=\s*(?P<value>[\d.eE+-]+)` |
   | 3 | 混合分隔符 | `epoch 3 \| loss: 0.1` | 组合模式1和2 |

5. **过滤规则**：
   - key 长度 1-32，纯字母数字下划线
   - value 必须是有效浮点数
   - 纯文本行（无数字）返回空 dict
   - 单个 value 出现多次时取最后一个

6. **自定义 pattern**：用户传入的正则优先级最高，使用命名捕获组

**测试覆盖**（test_parser.py）：
- `test_key_colon_value` — 基础 `key: value` 格式
- `test_key_equal_value` — 基础 `key=value` 格式
- `test_pipe_separated` — 管道分隔格式
- `test_dash_separated` — 横线分隔格式
- `test_multiple_metrics` — 一行多个指标
- `test_pure_text_ignored` — 纯文本返回空 dict
- `test_scientific_notation` — 科学计数法 `1.5e-4`
- `test_negative_values` — 负数值
- `test_custom_pattern` — 自定义正则
- `test_empty_input` — 空字符串
- `test_mixed_content` — 混合内容和指标
- `test_special_chars_in_key` — key 含特殊字符被过滤
- `test_unicode_text` — 中文等 Unicode 文本不被误解析

---

### 第三步：实现 writer.py（TensorBoard 写入层）

**设计要点**：

1. **单例管理器**：同一 `log_dir` 只创建一个 `SummaryWriter`，避免文件锁冲突

2. **核心类**：
   ```python
   class TBWriter:
       """Thread-safe TensorBoard writer wrapper with lifecycle management."""

       def __init__(self, log_dir: str = "runs"):
           self._log_dir = log_dir
           self._writer: Optional[SummaryWriter] = None
           self._global_step: int = 0
           self._lock = threading.Lock()
           self._active_writers: dict[str, "TBWriter"] = {}  # 单例缓存

       def get_writer(self) -> SummaryWriter:
           """Lazy-initialize and return the SummaryWriter."""

       def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
           """Log a scalar metric to TensorBoard."""

       def increment_step(self) -> int:
           """Increment and return the global step counter."""

       def close(self) -> None:
           """Flush and close the writer."""
   ```

3. **线程安全**：写入操作加锁，但解析不在锁内（解析无副作用）

4. **全局步数**：每个 writer 维护一个递增的 step，也可由调用方显式指定

---

### 第四步：实现 decorator.py（装饰器 + stdout 拦截）

**设计要点**：

1. **StreamProxy 类**：
   ```python
   class StreamProxy:
       """A file-like object that writes to both original stdout and an internal buffer."""

       def __init__(self, original: IO[str], buffer: IO[str]):
           self._original = original
           self._buffer = buffer

       def write(self, text: str) -> int:
           self._buffer.write(text)
           self._original.write(text)
           self._original.flush()  # 确保终端实时显示
           return len(text)

       def flush(self) -> None:
           self._original.flush()
           self._buffer.flush()

       def fileno(self) -> int:
           return self._original.fileno()
   ```

2. **tb_log 装饰器**：
   ```python
   def tb_log(
       log_dir: str = "runs",
       pattern: Optional[Union[str, re.Pattern]] = None,
       global_step: Optional[int] = None
   ) -> Callable[[F], F]:
       """Decorator that captures print output and logs metrics to TensorBoard."""
   ```

3. **装饰器工作流程**：
   ```
   @tb_log
   def train():
       ...

   # 等价于：
   def _wrapped(*args, **kwargs):
       # 1. 保存原始 stdout
       original_stdout = sys.stdout
       buffer = io.StringIO()

       # 2. 替换 stdout 为 StreamProxy
       sys.stdout = StreamProxy(original_stdout, buffer)

       try:
           # 3. 执行原函数
           result = func(*args, **kwargs)

           # 4. 读取 buffer 内容，逐行解析
           captured = buffer.getvalue()
           for line in captured.split('\n'):
               metrics = parse_print_output(line, pattern)
               for tag, value in metrics.items():
                   writer.log_scalar(tag, value, step)
                   step += 1
           return result
       except Exception:
           raise  # 不吞异常
       finally:
           # 5. 恢复 stdout，关闭 writer
           sys.stdout = original_stdout
           writer.close()
   ```

4. **tb_print 函数**：
   ```python
   def tb_print(
       tag: str,
       value: float,
       step: Optional[int] = None,
       log_dir: str = "runs",
       also_print: bool = True
   ) -> None:
       """Directly print and log a metric to TensorBoard."""
       if also_print:
           print(f"{tag}: {value}")
       writer = get_global_writer(log_dir)
       writer.log_scalar(tag, value, step=step)
   ```

5. **异常安全保证**：
   - `finally` 块中恢复 stdout
   - 原始异常用 `raise` 重新抛出，不包裹
   - writer.close() 在 finally 中调用

**测试覆盖**（test_decorator.py）：
- `test_decorator_captures_metrics` — 装饰器正确捕获并写入指标
- `test_decorator_preserves_output` — print 内容仍显示在终端
- `test_decorator_custom_pattern` — 自定义 pattern 生效
- `test_decorator_exception_safety` — 函数抛异常时 stdout 正确恢复
- `test_decorator_no_metrics_line` — 纯文本 print 不产生写入
- `test_decorator_nested_calls` — 嵌套函数调用不冲突
- `test_tb_print_direct` — tb_print 直接写入
- `test_decorator_step_increment` — step 正确递增
- `test_decorator_multiple_invocations` — 多次调用装饰器函数

---

### 第五步：编写 examples

1. **basic_demo.py**：
   ```python
   from printboard import tb_log
   import time, random

   @tb_log(log_dir="runs/basic_demo")
   def train():
       for epoch in range(10):
           loss = 1.0 / (epoch + 1) + random.uniform(-0.05, 0.05)
           acc = min(0.99, 0.5 + epoch * 0.05 + random.uniform(-0.02, 0.02))
           lr = 0.001 * (0.95 ** epoch)
           print(f"epoch {epoch} | loss: {loss:.4f} | acc: {acc:.4f} | lr: {lr:.6f}")
           time.sleep(0.1)
       print("Training completed!")

   if __name__ == "__main__":
       print("Starting training...")
       train()
       print("Done! Run: tensorboard --logdir=runs/basic_demo")
   ```

2. **custom_pattern.py**：演示自定义正则解析

---

### 第六步：README.md

按需求列出的 11 项内容完整撰写，英文、中文。

---

### 第七步：GitHub Actions CI

**`.github/workflows/ci.yml`**：

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Lint with flake8
        run: |
          flake8 printboard/ --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 printboard/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
      - name: Test with pytest
        run: |
          pytest tests/ -v --cov=printboard --cov-report=xml
      - name: Upload coverage
        uses: codecov/test-action@v3  # 可选
```

---

### 第八步：发布到 GitHub

- 创建远程仓库并推送
- 打初始 tag `v0.1.0`
- 添加 topics

---

## 四 · 执行顺序与轮次规划

按以下轮次执行，每轮完成后验证：

| 轮次 | 内容 | 验证方式 |
|------|------|----------|
| **Round 1** | 项目骨架 + git init + .gitignore + LICENSE + requirements.txt + pyproject.toml + setup.py | `python -c "import printboard"` 不报错 |
| **Round 2** | parser.py + test_parser.py | `pytest tests/test_parser.py -v` 全通过 |
| **Round 3** | writer.py | 手动创建 writer 写入并关闭，检查日志文件生成 |
| **Round 4** | decorator.py + tb_print + test_decorator.py | `pytest tests/test_decorator.py -v` 全通过 |
| **Round 5** | examples/ 两个示例 | 运行 basic_demo.py 观察输出和日志 |
| **Round 6** | README.md + .github/workflows/ci.yml | 检查 README 渲染，CI 配置语法 |
| **Round 7** | git commit → 创建分支 → PR → 合并 → 打 tag | 检查 git log 和分支结构 |

---

## 五 · 关键设计决策

### Q1: 为什么用函数结束后统一解析，而非实时逐行解析？

**答**：`print()` 在 Python 中是同步操作，但 stdout 可能有 buffering。在函数结束后读取 buffer 更可靠，且简化了线程安全设计。对于长时间运行的训练函数，用户可手动在 print 后 flush 或用 tb_print。

### Q2: 为什么 writer 不用全局单例？

**答**：不同装饰器可能配置不同的 `log_dir`。采用 `log_dir → TBWriter` 的缓存映射，同一目录共享 writer，不同目录隔离。

### Q3: 自定义 pattern 的优先级为什么最高？

**答**：用户显式指定的 pattern 表达的是精确意图，应覆盖所有内置启发式解析，避免意外匹配。

---

## 六 · 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| stdout 替换影响其他库的 logging | 仅在装饰器作用域内替换，函数结束后立即恢复 |
| 大量 print 导致性能下降 | 正则预编译 + 批量写入 + 仅解析含数字的行 |
| 多线程训练 stdout 竞争 | StreamProxy 的 write 使用 threading.Lock 保护 |
| torch/tensorboard 版本兼容性 | 仅使用 SummaryWriter 的公共 API，CI 覆盖多版本 |
