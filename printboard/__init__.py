"""PrintBoard - Zero-code decorator to bridge print() to TensorBoard."""

__version__ = "0.1.0"
__all__ = ["tb_log", "tb_print"]

# Lazy imports to allow incremental development
def __getattr__(name):
    if name == "tb_log":
        from printboard.decorator import tb_log
        return tb_log
    if name == "tb_print":
        from printboard.decorator import tb_print
        return tb_print
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
