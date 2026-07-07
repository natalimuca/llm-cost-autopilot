"""Build the classifier's labeled dataset from real, human-written prompts
instead of the template-generated seed set (see generate_dataset.py, which
is now a fallback/bootstrap only).

Source: databricks-dolly-15k (CC BY-SA 3.0), ~15k real instructions written
by Databricks employees, each pre-tagged with a task category. We map those
categories onto our 3 complexity tiers as a documented heuristic:

  tier 1 (simple)   <- open_qa, general_qa, closed_qa, information_extraction
                       Basic factual Q&A or extraction from provided context.
  tier 2 (moderate) <- classification, summarization
                       Structured analysis of provided content.
  tier 3 (complex)  <- creative_writing, brainstorming
                       Open-ended generation with no single correct answer.

This is a heuristic, not a per-example hand label -- category is a decent
proxy for complexity but isn't perfect (some "open_qa" rows are genuinely
hard, some "creative_writing" rows are one-liners). Spot-check a sample
before trusting it blindly; see the --sample-check flag.

Usage:
    python -m scripts.fetch_real_dataset
    python -m scripts.fetch_real_dataset --sample-check 15
"""
import argparse
import csv
import json
import random
import urllib.request
from pathlib import Path

random.seed(7)

RAW_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "databricks-dolly-15k.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "app" / "classifier" / "data" / "labeled_prompts.csv"

CATEGORY_TO_TIER = {
    "open_qa": 1,
    "general_qa": 1,
    "closed_qa": 1,
    "information_extraction": 1,
    "classification": 2,
    "summarization": 2,
    "creative_writing": 3,
    "brainstorming": 3,
}

PER_TIER_TARGET = 300


def download_if_missing() -> None:
    if RAW_PATH.exists():
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {RAW_URL} ...")
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)


def load_rows() -> list[dict]:
    rows = []
    with RAW_PATH.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_prompt(row: dict) -> str:
    instruction = row["instruction"].strip()
    context = row.get("context", "").strip()
    if context:
        return f"{instruction}\n\nContext: {context}"
    return instruction


def build_dataset(rows: list[dict]) -> list[tuple[str, int]]:
    by_tier: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for row in rows:
        tier = CATEGORY_TO_TIER.get(row["category"])
        if tier is None:
            continue
        by_tier[tier].append(to_prompt(row))

    dataset: list[tuple[str, int]] = []
    for tier, prompts in by_tier.items():
        random.shuffle(prompts)
        sample = prompts[:PER_TIER_TARGET]
        dataset.extend((p, tier) for p in sample)
    random.shuffle(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-check", type=int, default=0,
        help="Print N random (prompt, tier) pairs to manually spot-check the category->tier mapping.",
    )
    args = parser.parse_args()

    download_if_missing()
    rows = load_rows()
    dataset = build_dataset(rows)

    if args.sample_check:
        print(f"--- {args.sample_check} random samples for manual spot-check ---")
        for prompt, tier in random.sample(dataset, args.sample_check):
            print(f"[tier {tier}] {prompt[:200]!r}")
        print("---")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "tier"])
        writer.writerows(dataset)

    counts = {t: sum(1 for _, tier in dataset if tier == t) for t in (1, 2, 3)}
    print(f"Wrote {len(dataset)} real, human-written examples to {OUTPUT_PATH}")
    print(f"Per-tier counts: {counts}")


if __name__ == "__main__":
    main()
