import pytest

from app.models.providers.retry import with_retry


class FlakyError(Exception):
    pass


@pytest.mark.asyncio
async def test_with_retry_recovers_from_transient_failures():
    calls = {"count": 0}

    @with_retry(FlakyError, max_attempts=5, min_wait=0, max_wait=0)
    async def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise FlakyError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_with_retry_gives_up_after_max_attempts_and_reraises():
    calls = {"count": 0}

    @with_retry(FlakyError, max_attempts=3, min_wait=0, max_wait=0)
    async def always_fails():
        calls["count"] += 1
        raise FlakyError("permanent")

    with pytest.raises(FlakyError):
        await always_fails()
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_with_retry_does_not_catch_unrelated_exceptions():
    @with_retry(FlakyError, max_attempts=5, min_wait=0, max_wait=0)
    async def wrong_error():
        raise ValueError("not a transient provider error")

    with pytest.raises(ValueError):
        await wrong_error()
