"""Generate the seed labeled dataset for the complexity classifier.

This produces template-based examples, not a substitute for real usage data.
Ship V1 with this, then replace/augment rows in
app/classifier/data/labeled_prompts.csv with real hand-labeled prompts and
accumulated verifier failures (see Phase 3's feedback loop) as they come in.

Usage: python -m scripts.generate_dataset
"""
import csv
import random
from pathlib import Path

random.seed(7)

OUTPUT_PATH = Path(__file__).parent.parent / "app" / "classifier" / "data" / "labeled_prompts.csv"

# Tier 1 (simple): reformatting, extraction, basic Q&A from provided context.
TIER1_TEMPLATES = [
    "Extract the {field} from: '{context}'",
    "What is the capital of {place}?",
    "Reformat this as a bulleted list: {items}",
    "Convert '{value}' to {format}.",
    "What year did {event} happen?",
    "Find the phone number in this text: '{context}'",
    "Translate '{phrase}' into {language}.",
    "What is {a} plus {b}?",
    "Given the following text, what is the {field}? '{context}'",
    "Fix the spelling in this sentence: '{context}'",
]

# Tier 2 (moderate): summarization, classification, structured analysis.
TIER2_TEMPLATES = [
    "Summarize the following in 2-3 sentences: '{context}'",
    "Classify the sentiment of this review as positive, negative, or neutral: '{context}'",
    "Categorize this support ticket into billing, technical, or account: '{context}'",
    "Given this data, identify the trend: {data}",
    "List the key points from this article in bullet form: '{context}'",
    "Structure this information into a JSON object with fields {fields}: '{context}'",
    "Compare {a} and {b} in a short paragraph, covering at least two differences.",
    "Given the following table, summarize the top 3 rows: {data}",
    "Rewrite this paragraph to be more formal, keeping the meaning intact: '{context}'",
    "Group these items into categories: {items}",
]

# Tier 3 (complex): multi-step reasoning, creative generation, nuanced judgment.
TIER3_TEMPLATES = [
    "Analyze the trade-offs between {a} and {b} for a system that must handle {constraint}.",
    "Write a short creative story about {topic}, with a twist ending.",
    "Given these constraints — must {c1}, should {c2}, no more than {c3} — design an approach and justify each decision.",
    "Walk through, step by step, how you would debug a production incident where {scenario}.",
    "Critique this argument and identify any logical flaws: '{context}'",
    "First evaluate the pros and cons of {a}, then recommend {b} or an alternative, and justify your reasoning.",
    "Design a solution for {scenario} that balances cost, latency, and reliability. Explain your reasoning at each step.",
    "Given ambiguous requirements around {topic}, what clarifying questions would you ask, and how would that change your approach?",
    "Write a nuanced product decision memo about whether to {a} or {b}, weighing at least three factors.",
    "Explain the reasoning a senior engineer would use to choose between {a} and {b} under {constraint}.",
]

FILL = {
    "field": ["name and email", "order ID", "date", "total amount", "shipping address"],
    "context": [
        "Contact Jane Doe at jane@example.com.",
        "The order #48213 shipped on March 3rd for $129.99.",
        "Call us at (555) 012-3456 for support.",
        "The meeting is schedule for tommorow at 3pm in Conferance Room B.",
        "Revenue grew steadily through Q1 but dipped sharply in Q2 due to supply issues, then recovered by Q3.",
        "This product is a complete waste of money, though the packaging was nice.",
        "I can't log into my account and billing charged me twice this month.",
    ],
    "place": ["France", "Japan", "Brazil", "Kenya", "Canada"],
    "items": ["apples, bananas, cherries", "red, blue, green, yellow", "cat, dog, hamster, parrot"],
    "value": ["3.14159", "2024-03-01", "100 USD"],
    "format": ["two decimal places", "MM/DD/YYYY", "EUR"],
    "event": ["the moon landing", "the fall of the Berlin Wall", "the first iPhone launch"],
    "phrase": ["good morning", "thank you very much", "where is the train station"],
    "language": ["French", "Spanish", "German", "Japanese"],
    "a": ["REST", "microservices", "SQL databases", "renting", "in-house hosting", "sync processing"],
    "b": ["GraphQL", "a monolith", "NoSQL databases", "buying", "cloud hosting", "async processing"],
    "data": ["Jan: 100, Feb: 120, Mar: 90, Apr: 150", "Q1: $2M, Q2: $1.8M, Q3: $2.4M"],
    "fields": ["name, email, order_id", "title, author, year"],
    "constraint": ["10,000 requests/sec", "sub-100ms latency", "a hard budget cap", "strict uptime SLAs"],
    "topic": ["a robot learning to paint", "a lighthouse keeper's last night", "an AI questioning its purpose"],
    "c1": ["respond within 200ms", "support 1000 concurrent users", "stay under $500/month"],
    "c2": ["degrade gracefully under load", "log every failure", "be horizontally scalable"],
    "c3": ["3 external dependencies", "2 seconds of added latency", "$50 in monthly infra cost"],
    "scenario": [
        "checkout latency spiked 10x during a flash sale",
        "a memory leak caused nightly restarts",
        "one region silently dropped 5% of writes",
    ],
}


def _fill(template: str) -> str:
    out = template
    for key, options in FILL.items():
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", random.choice(options))
    return out


def generate(n_per_tier: int = 70) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for tier, templates in ((1, TIER1_TEMPLATES), (2, TIER2_TEMPLATES), (3, TIER3_TEMPLATES)):
        seen: set[str] = set()
        attempts = 0
        while len(seen) < n_per_tier and attempts < n_per_tier * 20:
            attempts += 1
            template = random.choice(templates)
            prompt = _fill(template)
            if prompt not in seen:
                seen.add(prompt)
                rows.append((prompt, tier))
    random.shuffle(rows)
    return rows


def main() -> None:
    rows = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "tier"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} labeled examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
