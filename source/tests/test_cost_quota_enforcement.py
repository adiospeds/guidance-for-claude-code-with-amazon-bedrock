# ABOUTME: Tests for cost-based quota enforcement (pricing utility + enforcement logic)
# ABOUTME: Validates cost calculation across model families and enforcement decisions

"""Tests for cost-based quota enforcement."""

import sys
from pathlib import Path

import pytest

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "deployment" / "infrastructure" / "lambda-functions"))

from shared.pricing import (
    DEFAULT_RATES,
    calculate_cost,
    cost_for_token_type,
    get_rates,
    resolve_model_family,
    resolve_rate_key,
)


class TestPricingUtility:
    """Tests for shared/pricing.py."""

    def test_sonnet_input_cost(self):
        """1M input tokens at Sonnet rates = $3.00."""
        cost = calculate_cost(1_000_000, 0, 0, 0, "sonnet")
        assert cost == pytest.approx(3.0)

    def test_sonnet_output_cost(self):
        """1M output tokens at Sonnet rates = $15.00."""
        cost = calculate_cost(0, 1_000_000, 0, 0, "sonnet")
        assert cost == pytest.approx(15.0)

    def test_sonnet_cache_read_cost(self):
        """1M cache read tokens at Sonnet rates = $0.30."""
        cost = calculate_cost(0, 0, 1_000_000, 0, "sonnet")
        assert cost == pytest.approx(0.3)

    def test_opus_input_cost(self):
        """1M input tokens at Opus rates = $5.00."""
        cost = calculate_cost(1_000_000, 0, 0, 0, "opus")
        assert cost == pytest.approx(5.0)

    def test_fable_input_cost(self):
        """1M input tokens at Fable rates = $10.00."""
        cost = calculate_cost(1_000_000, 0, 0, 0, "fable")
        assert cost == pytest.approx(10.0)

    def test_haiku_input_cost(self):
        """1M input tokens at Haiku rates = $1.00."""
        cost = calculate_cost(1_000_000, 0, 0, 0, "haiku")
        assert cost == pytest.approx(1.0)

    def test_mixed_tokens_sonnet(self):
        """Realistic mix: 500K input + 200K output + 10M cache reads."""
        cost = calculate_cost(500_000, 200_000, 10_000_000, 0, "sonnet")
        expected = (0.5 * 3.0) + (0.2 * 15.0) + (10.0 * 0.3)
        assert cost == pytest.approx(expected)  # $1.50 + $3.00 + $3.00 = $7.50

    def test_unknown_model_defaults_to_sonnet(self):
        """Unknown model family falls back to Sonnet rates."""
        cost = calculate_cost(1_000_000, 0, 0, 0, "unknown_model")
        assert cost == pytest.approx(3.0)

    def test_zero_tokens_zero_cost(self):
        """No tokens = no cost."""
        assert calculate_cost(0, 0, 0, 0) == 0.0


class TestCostForTokenType:
    """Tests for cost_for_token_type — the OTEL metric `type` → cost bridge.

    Regression: the quota_monitor cost calc used to look the metric `type` up
    in the rate table directly. The table key is `cache_write` but Claude Code
    emits `cacheCreation`, so every cache-write token was priced at $0, making
    the reported spend systematically lower than the AWS bill.
    """

    def test_cache_creation_is_not_free(self):
        """REGRESSION: `cacheCreation` tokens must be priced, not $0.

        Claude Code emits `type=cacheCreation` for cache writes. Bedrock bills
        these at 1.25x the input rate. A direct rate-table lookup returns $0.
        """
        cost = cost_for_token_type("cacheCreation", 1_000_000, "sonnet")
        assert cost == pytest.approx(3.75)  # 1.25 x $3.00 sonnet input
        assert cost > 0

    def test_cache_creation_opus(self):
        """`cacheCreation` at Opus rates = $6.25 per 1M (1.25 x $5 input)."""
        assert cost_for_token_type("cacheCreation", 1_000_000, "opus") == pytest.approx(6.25)

    def test_cache_read_camel_case(self):
        """`cacheRead` (Claude Code's camelCase) maps to the cache_read rate."""
        assert cost_for_token_type("cacheRead", 1_000_000, "sonnet") == pytest.approx(0.3)

    def test_input_output(self):
        """Plain input/output types price at their base rates."""
        assert cost_for_token_type("input", 1_000_000, "sonnet") == pytest.approx(3.0)
        assert cost_for_token_type("output", 1_000_000, "sonnet") == pytest.approx(15.0)

    def test_snake_case_variants(self):
        """snake_case type names are accepted too (non-Claude-Code paths)."""
        assert cost_for_token_type("cache_creation", 1_000_000, "sonnet") == pytest.approx(3.75)
        assert cost_for_token_type("cache_read", 1_000_000, "sonnet") == pytest.approx(0.3)

    def test_unknown_type_is_free(self):
        """Unrecognized/metadata token types carry no cost (not a crash)."""
        assert cost_for_token_type("bogus", 1_000_000, "sonnet") == 0.0

    def test_unknown_model_defaults_to_sonnet(self):
        """Unknown model family falls back to Sonnet rates."""
        assert cost_for_token_type("cacheCreation", 1_000_000, "mystery") == pytest.approx(3.75)


class TestResolveRateKey:
    """Tests for resolve_rate_key — metric `type` → rate-table key mapping."""

    def test_cache_creation_maps_to_cache_write(self):
        """The bug's root cause: cacheCreation must resolve to cache_write."""
        assert resolve_rate_key("cacheCreation") == "cache_write"

    def test_cache_read_maps_to_cache_read(self):
        assert resolve_rate_key("cacheRead") == "cache_read"

    def test_passthrough_types(self):
        assert resolve_rate_key("input") == "input"
        assert resolve_rate_key("output") == "output"

    def test_unknown_returns_none(self):
        assert resolve_rate_key("nonsense") is None


class TestModelResolution:
    """Tests for resolve_model_family."""

    def test_sonnet_cris(self):
        assert resolve_model_family("us.anthropic.claude-sonnet-4-6-v1") == "sonnet"

    def test_opus_cris(self):
        assert resolve_model_family("global.anthropic.claude-opus-4-7") == "opus"

    def test_haiku_cris(self):
        assert resolve_model_family("eu.anthropic.claude-haiku-4-5-20251001-v1:0") == "haiku"

    def test_fable_cris(self):
        assert resolve_model_family("us.anthropic.claude-fable-5") == "fable"

    def test_unknown_defaults_to_sonnet(self):
        assert resolve_model_family("some-random-model") == "sonnet"

    def test_empty_string(self):
        assert resolve_model_family("") == "sonnet"


class TestPricingOverride:
    """Tests for BEDROCK_PRICING_RATES_JSON env var override."""

    def test_override_merges_with_defaults(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_PRICING_RATES_JSON", '{"sonnet": {"input": 4.00}}')
        rates = get_rates()
        assert rates["sonnet"]["input"] == 4.00
        assert rates["sonnet"]["output"] == 15.00  # unchanged

    def test_invalid_json_uses_defaults(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_PRICING_RATES_JSON", "not valid json")
        rates = get_rates()
        assert rates["sonnet"]["input"] == 3.00

    def test_empty_env_uses_defaults(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_PRICING_RATES_JSON", "")
        rates = get_rates()
        assert rates == {k: dict(v) for k, v in DEFAULT_RATES.items()}


class TestCostEnforcementLogic:
    """Tests for quota_check cost enforcement decisions."""

    def _simulate_enforcement(self, usage: dict, policy: dict) -> dict:
        """Simulate the quota_check enforcement logic for cost."""
        monthly_cost = float(usage.get("cost_usd", 0))
        daily_cost = float(usage.get("daily_cost_usd", 0))
        monthly_cost_limit = float(policy.get("monthly_cost_limit", 0))
        daily_cost_limit = float(policy.get("daily_cost_limit", 0))

        if monthly_cost_limit > 0 and monthly_cost >= monthly_cost_limit:
            return {"allowed": False, "reason": "monthly_cost_exceeded"}
        if daily_cost_limit > 0 and daily_cost >= daily_cost_limit:
            return {"allowed": False, "reason": "daily_cost_exceeded"}
        return {"allowed": True, "reason": "within_budget"}

    def test_within_budget_allowed(self):
        result = self._simulate_enforcement(
            {"cost_usd": 30.0, "daily_cost_usd": 5.0}, {"monthly_cost_limit": 50.0, "daily_cost_limit": 10.0}
        )
        assert result["allowed"] is True

    def test_monthly_exceeded_blocked(self):
        result = self._simulate_enforcement(
            {"cost_usd": 55.0, "daily_cost_usd": 5.0}, {"monthly_cost_limit": 50.0, "daily_cost_limit": 10.0}
        )
        assert result["allowed"] is False
        assert result["reason"] == "monthly_cost_exceeded"

    def test_daily_exceeded_blocked(self):
        result = self._simulate_enforcement(
            {"cost_usd": 30.0, "daily_cost_usd": 12.0}, {"monthly_cost_limit": 50.0, "daily_cost_limit": 10.0}
        )
        assert result["allowed"] is False
        assert result["reason"] == "daily_cost_exceeded"

    def test_no_cost_limit_allows(self):
        """When no cost limit configured, cost enforcement is skipped."""
        result = self._simulate_enforcement(
            {"cost_usd": 999.0},
            {"monthly_cost_limit": 0},  # disabled
        )
        assert result["allowed"] is True

    def test_missing_cost_data_allows(self):
        """If cost_usd not in usage (old data), enforcement passes."""
        result = self._simulate_enforcement(
            {"total_tokens": 5000000},  # no cost_usd field
            {"monthly_cost_limit": 50.0},
        )
        assert result["allowed"] is True
