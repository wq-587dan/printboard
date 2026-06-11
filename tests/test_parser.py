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

    def test_no_space_after_colon(self):
        result = parse_print_output("loss:0.5")
        assert result == {"loss": 0.5}

    def test_space_before_colon(self):
        result = parse_print_output("loss : 0.5")
        assert result == {"loss": 0.5}


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

    def test_pipe_no_spaces(self):
        result = parse_print_output("|loss:0.1|acc:0.95")
        assert result["loss"] == 0.1
        assert result["acc"] == 0.95


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


class TestPositiveSign:
    """Test positive sign value parsing."""

    def test_positive_sign(self):
        result = parse_print_output("loss: +0.5")
        assert result == {"loss": 0.5}


class TestPercentage:
    """Test percentage value parsing."""

    def test_percentage_basic(self):
        result = parse_print_output("acc: 95%")
        assert result == {"acc": 95.0}

    def test_percentage_decimal(self):
        result = parse_print_output("acc: 95.5%")
        assert result == {"acc": 95.5}


class TestJsonLike:
    """Test JSON-like dict output parsing."""

    def test_json_single_quotes(self):
        result = parse_print_output("{'loss': 0.3456, 'acc': 0.92}")
        assert result["loss"] == 0.3456
        assert result["acc"] == 0.92

    def test_json_double_quotes(self):
        result = parse_print_output('{"loss": 0.3456, "acc": 0.92}')
        assert result["loss"] == 0.3456
        assert result["acc"] == 0.92


class TestSpecialValues:
    """Test nan and inf value handling."""

    def test_nan_value(self):
        result = parse_print_output("loss: nan")
        assert "loss" in result
        import math

        assert math.isnan(result["loss"])

    def test_inf_value(self):
        result = parse_print_output("loss: inf")
        assert "loss" in result
        assert result["loss"] == float("inf")

    def test_negative_inf_value(self):
        result = parse_print_output("loss: -inf")
        assert "loss" in result
        assert result["loss"] == float("-inf")


class TestTrailingComment:
    """Test lines with trailing comments."""

    def test_trailing_comment(self):
        result = parse_print_output("loss: 0.5 # training loss")
        assert result == {"loss": 0.5}


class TestCompactFormats:
    """Test compact formats without spaces."""

    def test_compact_no_spaces(self):
        result = parse_print_output("Loss:0.5,Acc:0.9")
        assert result["Loss"] == 0.5
        assert result["Acc"] == 0.9

    def test_compact_space_separated(self):
        result = parse_print_output("loss:0.5 acc:0.9")
        assert result["loss"] == 0.5
        assert result["acc"] == 0.9


class TestProgressBars:
    """Test progress bar format parsing."""

    def test_progress_bar_format(self):
        result = parse_print_output("1/10 [====] - loss: 0.5")
        assert result == {"loss": 0.5}


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
        assert result["loss"] == 0.3

    def test_key_too_long_filtered(self):
        long_key = "a" * 33
        result = parse_print_output(f"{long_key}: 1.0")
        assert long_key not in result

    def test_very_precise_value(self):
        result = parse_print_output("loss: 0.50000000001")
        assert abs(result["loss"] - 0.50000000001) < 1e-12

    def test_epoch_only_no_metrics(self):
        result = parse_print_output("Epoch 1/10")
        # "Epoch" is not a metric, and 1/10 is not key-value format
        assert result == {}


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

    def test_brackets_format(self):
        result = parse_print_output("[Epoch 1] Loss: 0.3456")
        assert result["Loss"] == 0.3456
