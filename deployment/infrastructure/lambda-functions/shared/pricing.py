# ABOUTME: Shared Bedrock pricing utility for quota cost calculations.
# ABOUTME: Maps model families to per-token-type rates ($/MTok) for cost-based enforcement.

"""
Bedrock pricing rates for cost-based quota enforcement.

Provides per-model-family, per-token-type pricing. Used by quota_monitor
to convert raw token counts into estimated USD spend.

IMPORTANT: These are estimates based on published Bedrock on-demand rates.
Actual costs may differ due to committed throughput, pricing changes, or
custom agreements. Use AWS Cost Explorer for billing truth.

Rates can be overridden via BEDROCK_PRICING_RATES_JSON env var.
"""

import json
import os

# OTEL `type` attribute values (as emitted by Claude Code and CoWork) mapped to
# the internal rate-table keys below. Claude Code emits camelCase `cacheRead` /
# `cacheCreation`; other paths may use snake_case. This map is the ONLY place
# that bridges metric type names → rate keys — never look a metric `type` up in
# the rate table directly (the table has no `cacheCreation` key, so a direct
# lookup silently prices cache-write tokens at $0). See resolve_rate_key().
TOKEN_TYPE_TO_RATE_KEY = {
    "input": "input",
    "output": "output",
    "cacheRead": "cache_read",
    "cache_read": "cache_read",
    "cacheCreation": "cache_write",
    "cache_creation": "cache_write",
    "cache_write": "cache_write",
}

# Rates below are STANDARD COMMERCIAL on-demand rates (us-*, eu-*, ap-* etc.).
# Cross-region inference (CRIS) is billed at the source region's on-demand rate
# with no routing premium, so these hold for `us.`/`eu.`/`global.` CRIS profiles.
# They do NOT hold for AWS GovCloud (higher) — for GovCloud or any region AWS
# prices differently, override the whole table via BEDROCK_PRICING_RATES_JSON.
#
# Per-model-family rates in USD per 1M tokens (as of June 2026)
# Source: https://aws.amazon.com/bedrock/pricing/
# cache_read = 0.1x input, cache_write = 1.25x input (5-minute prompt-cache TTL).
# NOTE: 1-hour-TTL cache writes are billed at 2.0x input, but the OTEL
# `cacheCreation` metric does not carry the TTL, so we cannot distinguish them
# and assume the default 5-minute rate.
DEFAULT_RATES = {
    "fable": {
        "input": 10.00,
        "output": 50.00,
        "cache_read": 1.00,
        "cache_write": 12.50,
    },
    "opus": {
        "input": 5.00,
        "output": 25.00,
        "cache_read": 0.50,
        "cache_write": 6.25,
    },
    "sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
    "haiku": {
        "input": 1.00,
        "output": 5.00,
        "cache_read": 0.10,
        "cache_write": 1.25,
    },
}

# Default model family when model cannot be resolved
DEFAULT_FAMILY = "sonnet"


def get_rates() -> dict:
    """Get pricing rates, with optional env var override.

    Override format (BEDROCK_PRICING_RATES_JSON):
    {"sonnet": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75}}
    """
    override = os.environ.get("BEDROCK_PRICING_RATES_JSON", "").strip()
    if override:
        try:
            custom = json.loads(override)
            merged = {k: dict(v) for k, v in DEFAULT_RATES.items()}
            for family, rates in custom.items():
                if family in merged:
                    merged[family].update(rates)
                else:
                    merged[family] = rates
            return merged
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    return {k: dict(v) for k, v in DEFAULT_RATES.items()}


def resolve_model_family(model_id: str) -> str:
    """Map a CRIS model ID to its pricing family.

    Examples:
        "us.anthropic.claude-sonnet-4-6-v1" → "sonnet"
        "global.anthropic.claude-opus-4-7"  → "opus"
        "eu.anthropic.claude-haiku-4-5-..."  → "haiku"
        "us.anthropic.claude-fable-5"       → "fable"
    """
    model_lower = model_id.lower()
    if "fable" in model_lower:
        return "fable"
    if "opus" in model_lower:
        return "opus"
    if "haiku" in model_lower:
        return "haiku"
    if "sonnet" in model_lower:
        return "sonnet"
    return DEFAULT_FAMILY


def calculate_cost(
    input_tokens: float,
    output_tokens: float,
    cache_read_tokens: float,
    cache_write_tokens: float = 0,
    model_family: str = DEFAULT_FAMILY,
    rates: dict | None = None,
) -> float:
    """Calculate estimated cost in USD from token counts.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cache_read_tokens: Number of cache read tokens
        cache_write_tokens: Number of cache write tokens (often 0 if not tracked)
        model_family: "fable", "opus", "sonnet", or "haiku"
        rates: Pricing rates dict (defaults to get_rates())

    Returns:
        Estimated cost in USD
    """
    if rates is None:
        rates = get_rates()

    family_rates = rates.get(model_family, rates.get(DEFAULT_FAMILY, {}))

    cost = (
        (input_tokens / 1_000_000) * family_rates.get("input", 3.0)
        + (output_tokens / 1_000_000) * family_rates.get("output", 15.0)
        + (cache_read_tokens / 1_000_000) * family_rates.get("cache_read", 0.3)
        + (cache_write_tokens / 1_000_000) * family_rates.get("cache_write", 3.75)
    )
    return cost


def resolve_rate_key(token_type: str) -> str | None:
    """Map an OTEL metric `type` value to a rate-table key.

    Returns None for token types that carry no price (unknown/metadata
    dimensions). Critically, `cacheCreation` → `cache_write` so cache-write
    tokens are not silently priced at $0.
    """
    return TOKEN_TYPE_TO_RATE_KEY.get(token_type)


def cost_for_token_type(
    token_type: str,
    tokens: float,
    model_family: str = DEFAULT_FAMILY,
    rates: dict | None = None,
) -> float:
    """Cost in USD for `tokens` of a single OTEL token `type`.

    Args:
        token_type: OTEL `type` value ("input", "output", "cacheRead",
            "cacheCreation", or their snake_case forms).
        tokens: Number of tokens of that type.
        model_family: "fable", "opus", "sonnet", or "haiku".
        rates: Pricing rates dict (defaults to get_rates()).

    Returns:
        Estimated cost in USD (0.0 for unrecognized token types).
    """
    rate_key = resolve_rate_key(token_type)
    if rate_key is None:
        return 0.0
    if rates is None:
        rates = get_rates()
    family_rates = rates.get(model_family, rates.get(DEFAULT_FAMILY, {}))
    return (tokens / 1_000_000) * family_rates.get(rate_key, 0.0)
