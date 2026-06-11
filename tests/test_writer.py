"""Tests for the TensorBoard writer wrapper."""

import os
import threading

from printboard.writer import TBWriter


class TestTBWriterBasic:
    """Test basic TBWriter functionality."""

    def test_creates_event_file(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)
        writer.log_scalar("test_metric", 1.0, step=0)
        writer.close()

        assert any("events" in f for f in os.listdir(log_dir))

    def test_log_scalar_with_explicit_step(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)
        writer.log_scalar("loss", 0.5, step=10)
        writer.log_scalar("acc", 0.9, step=20)
        writer.close()

        assert any("events" in f for f in os.listdir(log_dir))

    def test_global_step_auto_increment(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)

        step1 = writer.get_next_step()
        step2 = writer.get_next_step()
        step3 = writer.get_next_step()

        assert step1 == 0
        assert step2 == 1
        assert step3 == 2

        writer.close()

    def test_global_step_property(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)

        assert writer.global_step == 0
        writer.get_next_step()
        assert writer.global_step == 1
        writer.get_next_step()
        assert writer.global_step == 2

        writer.close()

    def test_reset_step(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)

        writer.get_next_step()
        writer.get_next_step()
        assert writer.global_step == 2

        writer.reset_step(100)
        assert writer.global_step == 100
        assert writer.get_next_step() == 100

        writer.close()


class TestTBWriterCaching:
    """Test writer caching by log directory."""

    def test_same_log_dir_shares_writer(self, tmp_path):
        log_dir = str(tmp_path)
        writer1 = TBWriter(log_dir=log_dir)
        writer2 = TBWriter(log_dir=log_dir)

        # Both should share the same underlying SummaryWriter.
        assert writer1.writer is writer2.writer

        writer1.close()

    def test_same_log_dir_shares_step_counter(self, tmp_path):
        log_dir = str(tmp_path)
        writer1 = TBWriter(log_dir=log_dir)
        writer2 = TBWriter(log_dir=log_dir)

        step1 = writer1.get_next_step()
        step2 = writer2.get_next_step()

        # Steps should be sequential, not both 0.
        assert step1 == 0
        assert step2 == 1

        writer1.close()

    def test_different_log_dirs_isolated(self, tmp_path):
        dir1 = str(tmp_path / "dir1")
        dir2 = str(tmp_path / "dir2")

        writer1 = TBWriter(log_dir=dir1)
        writer2 = TBWriter(log_dir=dir2)

        assert writer1.writer is not writer2.writer

        writer1.close()
        writer2.close()


class TestTBWriterThreadSafety:
    """Test thread safety of TBWriter."""

    def test_concurrent_log_scalar(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)
        errors = []

        def write_metrics(start, count):
            try:
                for i in range(count):
                    step = start * count + i
                    writer.log_scalar(f"thread_{start}", float(i), step=step)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_metrics, args=(t, 10)) for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        writer.close()

    def test_concurrent_step_increment(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)
        steps = []
        lock = threading.Lock()

        def increment_steps(count):
            local_steps = []
            for _ in range(count):
                s = writer.get_next_step()
                local_steps.append(s)
            with lock:
                steps.extend(local_steps)

        threads = [
            threading.Thread(target=increment_steps, args=(50,)) for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All steps should be unique and in range [0, 500).
        assert len(steps) == 500
        assert len(set(steps)) == 500
        assert min(steps) == 0
        assert max(steps) == 499

        writer.close()

    def test_concurrent_writer_creation(self, tmp_path):
        log_dir = str(tmp_path)
        writers = []
        errors = []

        def create_writer():
            try:
                w = TBWriter(log_dir=log_dir)
                writers.append(w)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_writer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(writers) == 10
        # All should share the same underlying writer.
        for w in writers[1:]:
            assert w.writer is writers[0].writer


class TestTBWriterLifecycle:
    """Test writer lifecycle management."""

    def test_close_removes_from_cache(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)

        assert log_dir in TBWriter._writers
        writer.close()
        assert log_dir not in TBWriter._writers

    def test_reset_clears_all(self, tmp_path):
        dir1 = str(tmp_path / "dir1")
        dir2 = str(tmp_path / "dir2")

        TBWriter(log_dir=dir1)
        TBWriter(log_dir=dir2)

        TBWriter.reset()

        assert len(TBWriter._writers) == 0

    def test_log_after_reset_creates_new_writer(self, tmp_path):
        log_dir = str(tmp_path)
        writer1 = TBWriter(log_dir=log_dir)
        old_writer = writer1.writer

        TBWriter.reset()

        writer2 = TBWriter(log_dir=log_dir)
        assert writer2.writer is not old_writer
        writer2.close()

    def test_writer_persists_across_calls(self, tmp_path):
        """Test that writer is NOT closed on every decorator-like call."""
        log_dir = str(tmp_path)

        w1 = TBWriter(log_dir=log_dir)
        w1.log_scalar("a", 1.0, step=0)

        w2 = TBWriter(log_dir=log_dir)
        # Should be the same SummaryWriter instance, not a new one.
        assert w1.writer is w2.writer
        w2.log_scalar("b", 2.0, step=1)

        w1.close()

    def test_multiple_log_scalar_calls(self, tmp_path):
        """Test that multiple log_scalar calls accumulate correctly."""
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)

        for i in range(10):
            writer.log_scalar("metric", float(i), step=i)

        writer.close()
        assert any("events" in f for f in os.listdir(log_dir))


class TestTBWriterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_log_scalar_with_negative_value(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)
        writer.log_scalar("negative_metric", -0.5, step=0)
        writer.close()

        assert any("events" in f for f in os.listdir(log_dir))

    def test_log_scalar_with_scientific_notation(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)
        writer.log_scalar("tiny_value", 1.5e-10, step=0)
        writer.log_scalar("huge_value", 1.2e10, step=1)
        writer.close()

        assert any("events" in f for f in os.listdir(log_dir))

    def test_log_scalar_with_special_tag_names(self, tmp_path):
        log_dir = str(tmp_path)
        writer = TBWriter(log_dir=log_dir)
        writer.log_scalar("train/loss", 0.5, step=0)
        writer.log_scalar("val/accuracy", 0.9, step=0)
        writer.close()

        assert any("events" in f for f in os.listdir(log_dir))

    def test_default_log_dir(self, tmp_path):
        """Test that default log_dir='runs' works."""
        writer = TBWriter(log_dir=str(tmp_path))
        writer.log_scalar("test", 1.0, step=0)
        writer.close()

        assert any("events" in f for f in os.listdir(str(tmp_path)))
