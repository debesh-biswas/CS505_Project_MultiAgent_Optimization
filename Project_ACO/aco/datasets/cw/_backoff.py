"""Exponential backoff retry helper for NVIDIA NIM HTTP 429 rate-limit errors."""

from __future__ import annotations

import random
import time

# NVIDIA NIM rate limit: 40 req/min → 60 s resets the full window.
# Start at 15 s (partial relief), double each retry, cap at 60 s.
_BASE_WAIT = 15.0   # seconds for first retry
_MULTIPLIER = 2.0
_CAP = 60.0         # full rate-limit window reset


def is_rate_limit_error(exc: BaseException) -> bool:
    """Return True when the exception is an HTTP 429 / rate-limit error."""
    name = type(exc).__name__
    msg = str(exc).lower()
    return (
        "RateLimitError" in name
        or "429" in str(exc)
        or "rate limit" in msg
        or "rate_limit" in msg
        or "too many requests" in msg
    )


def call_with_backoff(fn, *args, max_attempts: int = 9, **kwargs):
    """
    Call fn(*args, **kwargs) retrying up to max_attempts times on 429 errors.

    Back-off schedule (seconds, with ±20 % jitter, capped at 60 s):
      attempt 1 → immediate
      wait ~15 s → attempt 2
      wait ~30 s → attempt 3
      wait ~60 s → attempts 4-9   (full NVIDIA NIM 40 req/min window reset)

    Non-429 exceptions propagate immediately (no retry).
    Raises the last 429 exception when all attempts are exhausted.
    """
    wait = _BASE_WAIT
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:
            if not is_rate_limit_error(exc):
                raise
            if attempt == max_attempts:
                print(
                    f"  [backoff] rate-limit on attempt {attempt}/{max_attempts}; giving up."
                )
                raise
            jitter = random.uniform(0.8, 1.2)
            sleep_secs = min(wait * jitter, _CAP)
            print(
                f"  [backoff] rate-limit (attempt {attempt}/{max_attempts}); "
                f"retrying in {sleep_secs:.1f}s ..."
            )
            time.sleep(sleep_secs)
            wait = min(wait * _MULTIPLIER, _CAP)
