"""Retry pacing policy: pinned linear backoff vs shipped exponential default.

``agent.retry_backoff: linear`` in config.yaml (with optional
``agent.retry_backoff_seconds`` / ``agent.retry_backoff_max_seconds``) switches
mid-turn provider-retry waits from jittered exponential to deterministic
``min(delay * attempt, cap)``. Unset config keeps the shipped behavior
byte-identical. These tests pin the CONTRACT between the config stamps and the
computed delay — not any specific shipped default.
"""

import pytest

from agent import retry_utils
from agent.retry_utils import linear_backoff, retry_backoff_seconds


class _Agent:
    """Only the retry-pacing attributes the resolver reads."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


# --- linear_backoff: pure function contract ---------------------------------

@pytest.mark.parametrize("attempt,delay,cap,expected", [
    (1, 3.0, 30.0, 3.0),
    (2, 3.0, 30.0, 6.0),
    (10, 3.0, 30.0, 30.0),      # capped
    (4, 2.5, 100.0, 10.0),
    (1, 0.0, 30.0, 0.0),       # zero delay honored
    (3, 5.0, 0.0, 0.0),        # zero cap honored
])
def test_linear_backoff_values(attempt, delay, cap, expected):
    assert linear_backoff(attempt, delay=delay, max_delay=cap) == expected


def test_linear_backoff_deterministic():
    """No jitter: same inputs → identical outputs across calls."""
    first = [linear_backoff(a, delay=3.0, max_delay=30.0) for a in range(1, 8)]
    second = [linear_backoff(a, delay=3.0, max_delay=30.0) for a in range(1, 8)]
    assert first == second


def test_linear_backoff_rejects_zero_attempt_pace():
    """Attempt 0 is clamped to one delay unit — never a zero wait."""
    assert linear_backoff(0, delay=3.0, max_delay=30.0) == 3.0


# --- retry_backoff_seconds: policy resolution --------------------------------

def test_linear_policy_overrides_site_defaults():
    agent = _Agent(_retry_backoff_linear=True, _retry_backoff_seconds=3.0,
                   _retry_backoff_max_seconds=30.0)
    assert retry_backoff_seconds(agent, 2, base_delay=5.0, max_delay=120.0) == 6.0


@pytest.mark.real_retry_backoff
def test_default_agent_keeps_exponential():
    """No config stamps (fresh AIAgent) → jittered exponential site default."""
    agent = _Agent()
    wait = retry_backoff_seconds(agent, 1, base_delay=5.0, max_delay=120.0)
    # exponential: base 5.0 (+jitter in [0, 2.5]) — never the linear 3.0
    assert 5.0 <= wait <= 7.5


@pytest.mark.real_retry_backoff
def test_exponential_explicitly_configured_stays_exponential():
    agent = _Agent(_retry_backoff_linear=False)
    wait = retry_backoff_seconds(agent, 1, base_delay=5.0, max_delay=120.0)
    assert 5.0 <= wait <= 7.5


def test_linear_policy_missing_seconds_falls_back_to_defaults():
    """Pinned linear but knobs absent → 3.0/30.0 defaults, not a crash."""
    agent = _Agent(_retry_backoff_linear=True)
    assert retry_backoff_seconds(agent, 1, base_delay=5.0, max_delay=120.0) == 3.0
    assert retry_backoff_seconds(agent, 10, base_delay=5.0, max_delay=120.0) == 30.0


def test_cap_at_least_delay():
    agent = _Agent(_retry_backoff_linear=True, _retry_backoff_seconds=10.0,
                   _retry_backoff_max_seconds=5.0)  # misconfigured cap < delay
    # the resolver never widens a misconfiguration silently; linear_backoff itself
    # honours both, and attempt>=1 always yields >= delay*1 — pinned: cap wins at attempt 1
    assert retry_backoff_seconds(agent, 1, base_delay=5.0, max_delay=120.0) == 5.0
