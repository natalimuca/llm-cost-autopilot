"""Label the previously-excluded open_qa/general_qa Dolly prompts with GPT-4o
instead of dropping them or guessing from category alone.

Those two categories mix genuinely trivial factual questions ("What is
EFTPOS?") with genuinely more nuanced ones ("Why do people like dogs so
much?"), and Dolly's category tag alone can't tell the two apart -- but a
real model, judging each prompt individually, can. This adds real-world
coverage back to the classifier's training data instead of just narrowing
what it's tested against.

Requires OPENAI_API_KEY billing to be live (makes real GPT-4o calls).

Usage: python -m scripts.label_with_llm --n 400
"""
import argparse
import asyncio
import csv
import random
import re

from dotenv import load_dotenv

from app.models.interface import send_request
from scripts.fetch_real_dataset import OUTPUT_PATH, download_if_missing, load_rows, to_prompt

load_dotenv()
random.seed(11)

TARGET_CATEGORIES = {"open_qa", "general_qa"}

LABEL_PROMPT = """Classify the complexity of the following user prompt into exactly one tier:

Tier 1 (simple): a basic factual question or request with a short, direct, low-ambiguity answer. Little to no reasoning required.
Tier 2 (moderate): requires structured analysis -- classification, summarization, or synthesizing a few pieces of information -- but is still fairly mechanical.
Tier 3 (complex): requires multi-step reasoning, subjective or nuanced judgment, an opinion, or open-ended/creative generation.

Prompt: {prompt}

Respond with ONLY a single digit: 1, 2, or 3."""


def _parse_tier(text: str) -> int | None:
    match = re.search(r"[123]", text)
    return int(match.group()) if match else None


async def _label_one(semaphore: asyncio.Semaphore, prompt: str) -> tuple[str, int] | None:
    async with semaphore:
        try:
            response = await send_request(LABEL_PROMPT.format(prompt=prompt), "gpt-4o")
        except Exception as exc:  # noqa: BLE001 - labeling pass, keep going on individual failures
            print(f"[fail] {type(exc).__name__}: {exc}")
            return None
        tier = _parse_tier(response.text)
        if tier is None:
            print(f"[unparseable] {response.text!r}")
            return None
        return prompt, tier


async def main(n: int, concurrency: int) -> None:
    download_if_missing()
    rows = [r for r in load_rows() if r["category"] in TARGET_CATEGORIES]
    random.shuffle(rows)
    prompts = [to_prompt(r) for r in rows[:n]]

    semaphore = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*(_label_one(semaphore, p) for p in prompts))
    labeled = [r for r in results if r is not None]

    counts = {1: 0, 2: 0, 3: 0}
    for _, tier in labeled:
        counts[tier] += 1
    print(f"Labeled {len(labeled)}/{len(prompts)} prompts. Per-tier counts: {counts}")

    with OUTPUT_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(labeled)
    print(f"Appended to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args.n, args.concurrency))
