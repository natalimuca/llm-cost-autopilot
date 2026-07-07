"""Send a batch of diverse prompts through the running API to generate a
realistic cost-savings report and dashboard data for the portfolio writeup.

Usage: python -m scripts.load_test --n 500 --url http://localhost:8000
"""
import argparse
import asyncio
import random
import time

import httpx

from scripts.generate_dataset import generate


async def _send_one(client: httpx.AsyncClient, url: str, prompt: str) -> dict:
    start = time.perf_counter()
    resp = await client.post(
        f"{url}/v1/completions",
        json={"messages": [{"role": "user", "content": prompt}]},
        timeout=60.0,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    body = resp.json()
    body["wall_clock_seconds"] = elapsed
    return body


async def main(n: int, url: str, concurrency: int) -> None:
    pool = generate(n_per_tier=max(n // 3, 1))
    prompts = [p for p, _ in random.sample(pool, min(n, len(pool)))]

    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(client: httpx.AsyncClient, prompt: str) -> dict | None:
        async with semaphore:
            try:
                return await _send_one(client, url, prompt)
            except Exception as exc:  # noqa: BLE001 - load test, keep going
                print(f"[fail] {exc}")
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
