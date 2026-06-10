"""Tests for the print output parser."""

import re

from printboard.parser import parse_print_output


class TestKeyValueColon:
    """Test key: value format parsing."""

    def test_single_metric_colon(self):
        result = parse_print_output("loss: 0.3456")
        assert result == {"loss": 0.3456}

    def test_multiple_metrics_colon(self):
        result = parse_print_output("loss: 0.3456, acc: 0.92")
        assert result["loss"] == 0.3456
        assert result["acc"] == 0.92

    def test_metric_with_integer(self):
        result = parse_print_output("accuracy: 95")
        assert result == {"accuracy": 95.0}


class TestKeyValueEqual:
    """Test key=value format parsing."""

    def test_single_metric_equal(self):
        result = parse_print_output("loss=0.3456")
        assert result == {"loss": 0.3456}

    def test_multiple_metrics_equal(self):
        result = parse_print_output("loss=0.3456,acc=0.92")
        assert result["loss"] == 0.3456
        assert result["acc"] == 0.92

    def test_metric_equal_with_spaces(self):
        result = parse_print_output("loss = 0.3456")
        assert result == {"loss": 0.3456}


class TestPipeSeparated:
    """Test pipe-separated format parsing."""

    def test_pipe_separated_basic(self):
        result = parse_print_output("epoch 3 | loss: 0.3456 | acc: 0.92")
        assert result["loss"] == 0.3456
        assert result["acc"] == 0.92

    def test_pipe_separated_multiple(self):
        result = parse_print_output("loss: 0.1 | acc: 0.95 | lr: 0.001")
        assert result["loss"] == 0.1
        assert result["acc"] == 0.95
        assert result["lr"] == 0.001


class TestDashSeparated:
    """Test dash-separated format parsing."""

    def test_dash_separated(self):
        result = parse_print_output("Step 100 - loss: 0.3456")
        assert result == {"loss": 0.3456}

    def test_double_dash_separated(self):
        result = parse_print_output("Epoch 5 -- acc: 0.93")
        assert result == {"acc": 0.93}


class TestScientificNotation:
    """Test scientific notation value parsing."""

    def test_positive_exponent(self):
        result = parse_print_output("lr: 1.5e-4")
        assert result == {"lr": 1.5e-4}

    def test_negative_exponent(self):
        result = parse_print_output("loss: 2.5E-3")
        assert result == {"loss": 0.0025}


class TestNegativeValues:
    """Test negative value parsing."""

    def test_negative_metric(self):
        result = parse_print_output("loss: -0.5")
        assert result == {"loss": -0.5}


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_input(self):
        result = parse_print_output("")
        assert result == {}

    def test_none_like_input(self):
        result = parse_print_output("   ")
        assert result == {}

    def test_pure_text_ignored(self):
        result = parse_print_output("Training started...")
        assert result == {}

    def test_chinese_text_ignored(self):
        result = parse_print_output("训练开始，请稍候...")
        assert result == {}

    def test_mixed_content(self):
        result = parse_print_output("Epoch 3/10 - loss: 0.1234, val_loss: 0.1567")
        assert result["loss"] == 0.1234
        assert result["val_loss"] == 0.1567

    def test_duplicate_keys_takes_last(self):
        result = parse_print_output("loss: 0.5, loss: 0.3")
        # Last value wins for same key in general pattern.
        assert result["loss"] == 0.3

    def test_key_too_long_filtered(self):
        long_key = "a" * 33
        result = parse_print_output(f"{long_key}: 1.0")
        # Keys longer than 32 chars should be filtered.
        assert long_key not in result


class TestCustomPattern:
    """Test custom regex pattern parsing."""

    def test_custom_pattern_basic(self):
        pattern = re.compile(r"(?P<loss>[\d.]+)/(?P<acc>[\d.]+)")
        result = parse_print_output("0.3456/0.9200", custom_pattern=pattern)
        assert result["loss"] == 0.3456
        assert result["acc"] == 0.92

    def test_custom_pattern_overrides_builtin(self):
        pattern = re.compile(r"(?P<mynum>\d+)")
        result = parse_print_output("loss: 5", custom_pattern=pattern)
        assert result == {"mynum": 5.0}

    def test_custom_pattern_no_match(self):
        pattern = re.compile(r"(?P<only_float>\d+\.\d+)")
        result = parse_print_output("no float here", custom_pattern=pattern)
        assert result == {}

    def test_custom_pattern_named_groups_only(self):
        pattern = re.compile(r"([\d.]+):([\d.]+)")
        result = parse_print_output("0.1:0.2", custom_pattern=pattern)
        # Non-named groups are ignored.
        assert result == {}


class TestMixedFormats:
    """Test mixed format scenarios."""

    def test_colon_and_equal_in_same_line(self):
        result = parse_print_output("loss: 0.5, best=0.3")
        assert result["loss"] == 0.5
        assert result["best"] == 0.3

    def test_pipe_and_general_mixed(self):
        result = parse_print_output("status: ok | loss: 0.1")
        assert result["loss"] == 0.1
