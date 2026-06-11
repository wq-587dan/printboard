"""Tests for utility functions."""

from printboard.utils import is_metric_value, is_valid_metric_name, sanitize_metric_name


class TestIsValidMetricName:
    def test_valid_simple(self):
        assert is_valid_metric_name("loss") is True

    def test_valid_with_underscore(self):
        assert is_valid_metric_name("train_loss") is True

    def test_valid_with_slash(self):
        assert is_valid_metric_name("train/loss") is True

    def test_valid_with_hyphen(self):
        assert is_valid_metric_name("my-metric") is True

    def test_invalid_empty(self):
        assert is_valid_metric_name("") is False

    def test_invalid_spaces(self):
        assert is_valid_metric_name("my metric") is False

    def test_invalid_too_long(self):
        assert is_valid_metric_name("a" * 33, max_length=32) is False

    def test_valid_at_max_length(self):
        assert is_valid_metric_name("a" * 32, max_length=32) is True


class TestIsMetricValue:
    def test_integer(self):
        assert is_metric_value("42") is True

    def test_float(self):
        assert is_metric_value("0.5") is True

    def test_scientific(self):
        assert is_metric_value("1.5e-4") is True

    def test_negative(self):
        assert is_metric_value("-0.5") is True

    def test_negative_exponent(self):
        assert is_metric_value("1.5E-10") is True

    def test_non_numeric(self):
        assert is_metric_value("abc") is False

    def test_empty(self):
        assert is_metric_value("") is False

    def test_mixed(self):
        assert is_metric_value("0.5abc") is False


class TestSanitizeMetricName:
    def test_already_clean(self):
        assert sanitize_metric_name("loss") == "loss"

    def test_replaces_spaces(self):
        assert sanitize_metric_name("my metric") == "my_metric"

    def test_truncates(self):
        assert len(sanitize_metric_name("a" * 100, max_length=10)) == 10

    def test_replaces_special_chars(self):
        result = sanitize_metric_name("loss@!#$%")
        assert "@" not in result
        assert result == "loss_____"
