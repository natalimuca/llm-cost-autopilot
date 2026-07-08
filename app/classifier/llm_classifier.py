"""LLM-based complexity classifier: gpt-4o-mini judges each prompt directly
instead of matching TF-IDF vocabulary against a trained model.

Why this exists: the TF-IDF classifier (classifier.py) plateaus around
63-65% accuracy on real, diverse prompts because tier is a *semantic*
judgment, not a lexical one -- "What is EFTPOS?" (tier 1) and "Why do people
like dogs so much?" (tier 2/3) use similarly plain, short vocabulary, so
bag-of-words features can't tell them apart. An LLM can, because it
actually understands the question. This is the "real fix" flagged
throughout classifier.py/train.py's docstrings once cloud billing was live.

Cost/latency tradeoff: this adds one extra gpt-4o-mini call to every
request just to decide routing. At gpt-4o-mini's pricing that's a small
fraction of a cent per request -- cheap enough that it doesn't meaningfully
erode the cost savings from routing itself.
"""
import json
import re

from app.models.interface import send_request

CLASSIFIER_MODEL = "gpt-4o-mini"

CLASSIFY_PROMPT = """You are routing a user's prompt to the cheapest AI model that can handle it well. Classify the prompt's complexity into exactly one tier:

Tier 1 (simple): basic factual questions, straightforward extraction, or reformatting. Short, direct, low-ambiguity answers.
Examples: "What is EFTPOS?", "Extract the founder's name from this text: ...", "What is 12 + 7?"

Tier 2 (moderate): structured analysis such as classification, summarization, or synthesizing a few pieces of information. Still fairly mechanical.
Examples: "Classify each of these as fruit or vegetable: apple, carrot", "Summarize this passage in two sentences: ..."

Tier 3 (complex): multi-step reasoning, subjective or nuanced judgment, creative generation, or open-ended brainstorming.
Examples: "Write a short story about a robot learning to paint", "Analyze the trade-offs between X and Y and recommend one", "Brainstorm five ideas for a coffee shop name"

Prompt to classify: {prompt}

Respond with ONLY a JSON object, no other text: {{"tier": <1, 2, or 3>, "confidence": <0.0 to 1.0>}}"""


def _parse(text: str) -> tuple[int, float]:
    try:
        data = json.loads(text)
        tier = int(data["tier"])
        confidence = float(data["confidence"])
        if tier in (1, 2, 3):
            return tier, max(0.0, min(1.0, confidence))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Fallback for a non-JSON reply: grab the first digit 1-3, default confidence.
    match = re.search(r"[123]", text)
    return (int(match.group()) if match else 2), 0.5


async def classify(prompt: str) -> tuple[int, float]:
    """Returns (tier, confidence) where tier is 1 (simple), 2 (moderate), or 3 (complex)."""
    response = await send_request(CLASSIFY_PROMPT.format(prompt=prompt), CLASSIFIER_MODEL)
    return _parse(response.text)
