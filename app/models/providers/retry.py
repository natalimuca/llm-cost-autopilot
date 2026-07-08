"""Shared retry policy for transient provider errors.

Not a hypothetical concern: scripts/label_with_llm.py's first run lost
115/400 calls to OpenAI 429s (see README) because nothing retried a rate
limit -- it just gave up on that single call. Every provider's send()
method gets this policy so classification, completion, and verification
calls all benefit without each provider re-implementing backoff.
"""
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def with_retry(
    *exception_types: type[Exception],
    max_attempts: int = 5,
    min_wait: float = 1,
    max_wait: float = 20,
):
    return retry(
        retry=retry_if_exception_type(exception_types),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    )
