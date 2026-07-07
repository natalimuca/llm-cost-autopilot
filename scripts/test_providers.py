"""Send the same set of prompts to every model in the registry and log
outputs, cost, and latency. Baseline data for the routing logic.

Usage: python -m scripts.test_providers
"""
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from app.models.interface import send_request
from app.models.registry import MODEL_REGISTRY

load_dotenv()

PROMPTS = [
    "Extract the name and email from: 'Contact Jane Doe at jane@example.com.'",
    "Reformat this as a bulleted list: apples, bananas, cherries",
    "What is the capital of France?",
    "Summarize in one sentence: The quick brown fox jumps over the lazy dog repeatedly during a long afternoon in the meadow.",
    "Classify the sentiment of this review as positive, negative, or neutral: 'The food was cold but the service was excellent.'",
    "Compare and contrast REST and GraphQL APIs in a short paragraph.",
    "Write a short creative story about a robot learning to paint.",
    "Given this data, analyze the trend: Jan: 100, Feb: 120, Mar: 90, Apr: 150.",
    "What are three constraints to consider when designing a rate limiter?",
    "Translate 'good morning' into French, Spanish, and German.",
]

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "baseline_results.json"


async def main() -> None:
    results = []
    for model_name in MODEL_REGISTRY:
        for i, prompt in enumerate(PROMPTS):
            try:
                response = await send_request(prompt, model_name)
                results.append(
                    {
                        "model": model_name,
                        "prompt_index": i,
                        "prompt": prompt,
                        "output": response.text,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "latency_seconds": response.latency_seconds,
                        "cost_usd": response.cost_usd,
                    }
                )
                print(f"[ok] {model_name} prompt#{i} cost=${response.cost_usd:.6f} latency={response.latency_seconds:.2f}s")
            except Exception as exc:  # noqa: BLE001 - baseline script, log and continue
                print(f"[fail] {model_name} prompt#{i}: {exc}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
