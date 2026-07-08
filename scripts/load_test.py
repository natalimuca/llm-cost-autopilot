"""Send a batch of diverse, real prompts through the running API to generate
a realistic cost-savings report and dashboard data for the portfolio writeup.

Draws from the same real Dolly-15k pool as the classifier's training data
(scripts.fetch_real_dataset), but across *all* of Dolly's categories, not
just the ones we trained on -- this deliberately includes prompt types the
classifier has never seen (e.g. context-free trivia) so the load test
reflects genuinely diverse real traffic, not just the easy cases.

Usage: python -m scripts.load_test --n 500 --url http://localhost:8000
"""
import argparse
import asyncio
import random
import time

import httpx

from scripts.fetch_real_dataset import download_if_missing, load_rows, to_prompt


async def _send_one(client: httpx.AsyncClient, url: str, prompt: str) -> dict:
    start = time.perf_counter()
    resp = await client.post(
        f"{url}/v1/completions",
        json={"messages": [{"role": "user", "content": prompt}]},
        timeout=120.0,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    body = resp.json()
    body["wall_clock_seconds"] = elapsed
    return body


def _load_real_prompts(n: int) -> list[str]:
    download_if_missing()
    rows = load_rows()
    random.shuffle(rows)
    return [to_prompt(row) for row in rows[:n]]


async def main(n: int, url: str, concurrency: int) -> None:
    prompts = _load_real_prompts(n)

    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(client: httpx.AsyncClient, prompt: str) -> dict | None:
        async with semaphore:
            try:
                return await _send_one(client, url, prompt)
            except Exception as exc:  # noqa: BLE001 - load test, keep going
                print(f"[fail] {type(exc).__name__}: {exc}")
                return None

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(_bounded(client, p) for p in prompts))

    ok = [r for r in results if r is not None]
    total_cost = sum(r["cost_usd"] for r in ok)
    print(f"Sent {len(prompts)} prompts, {len(ok)} succeeded, {len(prompts) - len(ok)} failed.")
    print(f"Total cost this run: ${total_cost:.4f}")
    print(f"Fetch the full savings report from GET {url}/v1/stats")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args.n, args.url, args.concurrency))
