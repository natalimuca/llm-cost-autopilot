# LLM Cost Autopilot

![CI](https://github.com/natalimuca/llm-cost-autopilot/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

A routing layer that sits in front of OpenAI, Anthropic, and a local Ollama
model, scores each incoming prompt's complexity, sends it to the cheapest
model likely to handle it well, and continuously verifies that the routing
decision was actually correct — escalating and self-correcting when it
wasn't.

**Live case-study site:** https://natalimuca.github.io/llm-cost-autopilot/
**Repository:** https://github.com/natalimuca/llm-cost-autopilot

> **Headline result:** 150 real, diverse prompts, live billing, no synthetic
> data — **68.9% cheaper** than sending everything to gpt-4o. See
> [Results](#results) for the full breakdown, and
> [The Classifier](#the-classifier-two-backends-one-honest-experiment) for
> how that number was earned rather than assumed.

## Table of contents

- [Why this exists](#why-this-exists)
- [Results](#results)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [The classifier: two backends, one honest experiment](#the-classifier-two-backends-one-honest-experiment)
- [Setup](#setup)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Testing & CI](#testing--ci)
- [Observability](#observability)
- [The feedback loop](#the-feedback-loop)
- [Known limitations & roadmap](#known-limitations--roadmap)
- [License](#license)

## Why this exists

Teams that put LLMs into production tend to default to the same model for
every request, because routing is extra engineering work and getting it
wrong is embarrassing. The result is systematic overspend: a one-line
extraction task and a nuanced reasoning task cost the same, even though one
of them would have been handled just as well — and just as correctly — by a
model priced at a fraction of the cost.

This project is a working answer to "how would you actually build a cost
router, not just argue that one should exist": classify complexity, route
by tier, verify the decision after the fact with a stronger model acting as
judge, auto-escalate on a real quality miss, and feed those misses back into
the classifier so it keeps improving. Every number quoted below came from
real API calls against live OpenAI and Anthropic billing, not a benchmark
or a projection.

## Results

Measured against `GET /v1/stats` after `scripts/load_test.py --n 150` ran
against genuinely diverse real prompts (see
[The classifier](#the-classifier-two-backends-one-honest-experiment) for
where they came from):

| Metric | Value |
|---|---|
| Requests | 153 |
| Cost if everything went to gpt-4o | $0.4450 |
| Actual cost, routed | $0.1384 |
| **Cost savings** | **68.9%** ($0.3065 saved) |
| Routing split | 53.6% free local Llama · 28.1% gpt-4o-mini · 18.3% top-tier |
| Avg. verifier quality score | 4.15 / 5 (scored by an independent LLM judge) |
| Escalation rate | 15.0% — real quality misses caught and auto-corrected |

The escalation rate isn't a defect to explain away: routing 54% of traffic
to a free, genuinely weaker local model means that model gets more prompts
wrong, and the verifier catching and fixing those in the background is the
entire point of the architecture. See [Results](#results) →
[The classifier](#the-classifier-two-backends-one-honest-experiment) for the
experiment that got the routing decision itself this accurate in the first
place, and `docs/index.html` for the full narrative with figures.

Re-run this yourself any time and the exact number will move with your
prompt mix — that's why it isn't hardcoded anywhere in the code.

## Architecture

```
                        ┌─────────────────────┐
  client ── POST ──────►│   FastAPI  /v1/*     │
             /v1/       │  (app/api/routes.py) │
             completions└──────────┬──────────┘
                                   │
                     1. classify   ▼
                 ┌─────────────────────────────┐
                 │  Complexity classifier       │  app/classifier/
                 │  gpt-4o-mini judges the prompt│
                 │  directly (TF-IDF fallback     │
                 │  available offline)            │
                 └──────────────┬───────────────┘
                                 │ tier 1/2/3
                     2. route    ▼
                 ┌─────────────────────────────┐
                 │  Router (config/routing.yaml)│  app/router/
                 │  + latency-aware reassignment │
                 └──────────────┬───────────────┘
                                 │ model name
                     3. call     ▼
                 ┌─────────────────────────────┐
                 │  Unified model interface     │  app/models/
                 │  send_request() ──► OpenAI /  │
                 │                     Anthropic /│
                 │                     Ollama     │
                 └──────────────┬───────────────┘
                                 │ Response (text, cost, tokens, latency)
                     4. respond  ▼
                          back to client
                                 │
                     5. log      ▼
                 ┌─────────────────────────────┐
                 │  SQLite audit trail          │  app/db/
                 │  + structured JSON log line   │  app/logging_config.py
                 └──────────────┬───────────────┘
                                 │
                6. queue async   ▼
                 ┌─────────────────────────────┐
                 │  Verifier worker (in-process  │  app/verifier/
                 │  asyncio queue): judge model   │
                 │  re-answers, scores agreement, │
                 │  flags routing failures,       │
                 │  auto-escalates                │
                 └──────────────┬───────────────┘
                                 │ routing failures
                7. feedback loop ▼
                 ┌─────────────────────────────┐
                 │  scripts/retrain_from_        │
                 │  feedback.py appends failures │
                 │  to the training set and      │
                 │  retrains the classifier       │
                 └─────────────────────────────┘
                                 │
                     dashboard   ▼
                 Streamlit reads app/db straight from SQLite
```

The verifier runs as an **in-process asyncio task**, not a separate
container — the queue is in-memory, so a real "background worker" service
would need a broker (Redis/Celery) that this project intentionally skips to
stay in scope. See `app/verifier/worker.py` for the reasoning.

## Project layout

```
app/
  models/       Phase 1 — ModelConfig registry, provider adapters (with retry/backoff), send_request()
  classifier/   Phase 2 — llm_classifier.py (default) + classifier.py/train.py (TF-IDF fallback)
  router/       Phase 2 — tier -> model mapping (config/routing.yaml), latency-aware reassignment
  verifier/     Phase 3 — LLM-as-judge scoring, async worker, auto-escalation
  db/           Phase 4 — SQLite schema + logging/query helpers
  dashboard/    Phase 4 — Streamlit cost/quality dashboard
  api/          Phase 5 — FastAPI routes + request/response schemas
  logging_config.py  structured JSON request logs (alongside SQLite)
  main.py       Phase 5 — FastAPI app, lifespan (DB init + worker startup)
scripts/
  fetch_real_dataset.py     TF-IDF fallback's labeled dataset: real prompts from Dolly-15k (see below)
  label_with_llm.py         GPT-4o labeling pass for categories fetch_real_dataset.py excludes
  generate_dataset.py       template-generated fallback/bootstrap dataset (Phase 2)
  test_providers.py         Phase 1 baseline: same prompts across every model
  retrain_from_feedback.py  Phase 3.4 flywheel: failures -> retrain
  load_test.py              Phase 6 load test against a running API (real Dolly prompts)
config/routing.yaml         tier -> model map, editable via PUT /v1/routing-config
docs/index.html              self-contained project showcase / case-study site
tests/                       pytest unit tests (testpaths scoped via pytest.ini so
                              `pytest` never tries to collect scripts/*.py)
```

## The classifier: two backends, one honest experiment

`config/routing.yaml`'s `classifier_backend` picks which one routes traffic:

- **`llm` (default, needs `OPENAI_API_KEY` billing)** — `app/classifier/llm_classifier.py`
  asks gpt-4o-mini to judge each prompt's tier directly via a few-shot
  prompt. Adds one small extra call per request; costs a small fraction of
  a cent.
- **`tfidf` (free, offline, no API key needed)** — `app/classifier/classifier.py`,
  TF-IDF + logistic regression trained on `app/classifier/data/labeled_prompts.csv`.
  Use this for CI, local dev without billing, or as a fallback.

### Why both exist — the experiment that led here

The training data started as 900 real, human-written prompts sampled from
[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
(CC BY-SA 3.0), not templates, with Dolly's task categories mapped onto our
3 tiers as a heuristic (`scripts/fetch_real_dataset.py`).

| Step | What changed | TF-IDF held-out accuracy |
|---|---|---|
| 1 | Template-generated synthetic prompts | 97.6% (looked great, meant nothing) |
| 2 | Real Dolly-15k prompts, `open_qa`/`general_qa` included in tier 1 | 63.9% |
| 3 | Dropped those two categories — narrower but cleaner labels | **84.4%** |
| 4 | Added them back with real per-example GPT-4o labels instead of a category guess | 64.6% |

Step 3 looked like a win, but step 4 revealed why it wasn't a real fix:
accuracy *dropped* back down even with objectively better labels, because
"What is EFTPOS?" (tier 1) and "Why do people like dogs so much?" (tier 2/3)
use similarly plain, short vocabulary — the complexity difference is
*semantic*, not lexical, and no amount of TF-IDF feature engineering can see
that distinction. That's a real ceiling of bag-of-words classification on
this task, not a data quality problem.

**That's the actual justification for the `llm` backend.** Once cloud
billing was available, switching the classifier itself to gpt-4o-mini
(which understands the prompt instead of pattern-matching its vocabulary)
directly targeted the gap the experiment proved existed:

| Metric | TF-IDF backend | LLM backend |
|---|---|---|
| Real load test (150 diverse prompts) cost savings | 28.8% | **68.9%** |
| Routing distribution | 58% to gpt-4o | 54% local/free, 28% mid-tier, 18% gpt-4o |
| Avg verifier quality score | 4.66/5 | 4.15/5 |
| Escalation rate | 2.0% | 15.0% |

The lower avg quality score and higher escalation rate under the `llm`
backend aren't regressions — they're the system routing far more
aggressively to a genuinely weaker free/cheap model (local Llama went from
5% to 54% of traffic), which naturally gets more prompts wrong, and the
verifier safety net catching and auto-correcting those in the background.
That's the entire point of the escalation loop working as designed, not a
side effect to explain away.

These numbers were measured with tier 3 → gpt-4o. `config/routing.yaml`
now maps tier 3 → **claude-sonnet** instead, so the live system is a real
three-provider mix (free local Llama, OpenAI mid-tier, Anthropic top-tier)
rather than OpenAI-only with Anthropic registered but unused.

Run `python -m scripts.fetch_real_dataset --sample-check 15` to spot-check
the Dolly category→tier mapping, or `python -m scripts.label_with_llm --n 400`
to re-run the GPT-4o labeling pass (note: subject to your account's
tokens-per-minute rate limit — ~285/400 succeeded at concurrency 10 on a
low-tier account, the rest hit 429s and were skipped, not silently dropped).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv\Scripts\Activate.ps1 on PowerShell
pip install -r requirements.txt
cp .env.example .env          # fill in OPENAI_API_KEY / ANTHROPIC_API_KEY
```

Local model: install [Ollama](https://ollama.com) and pull a model —
`ollama pull llama3.1:8b` — or swap `OLLAMA_MODEL` / the `llama-local` entry
in `app/models/registry.py` for whatever you have.

### 1. Validate the provider abstraction (Phase 1)

```bash
python -m scripts.test_providers
```

Sends the same 10 prompts to every model in the registry and writes
`data/baseline_results.json` with cost, latency, and output for each —
your first sanity check that all three providers are wired correctly.

### 2. Set up the classifier (Phase 2)

The default `classifier_backend: llm` in `config/routing.yaml` needs no
training step at all — it calls gpt-4o-mini directly. If you want the free
offline `tfidf` fallback available too (e.g. for CI or no-billing dev):

```bash
python -m scripts.fetch_real_dataset
python -m app.classifier.train
```

This downloads real, human-written prompts (see
[The classifier](#the-classifier-two-backends-one-honest-experiment) above)
and trains on them. If you'd rather bootstrap quickly with the older
template-generated set instead (faster, but inflated accuracy that doesn't
reflect real traffic), use `python -m scripts.generate_dataset` instead.
Either way, the feedback loop appends real verifier-caught routing failures
to whichever dataset you're using.

### 3. Run the API + dashboard

```bash
uvicorn app.main:app --reload
streamlit run app/dashboard/streamlit_app.py   # separate terminal
```

or via Docker:

```bash
docker-compose up --build
```

### 4. Send a request

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarize this in one sentence: ..."}]}'
```

### 5. Load test + generate your own numbers (Phase 6)

```bash
python -m scripts.load_test --n 150 --url http://localhost:8000
curl http://localhost:8000/v1/stats
```

Screenshot the Streamlit dashboard and the `/v1/stats` payload — those are
the artifacts for a writeup, and `docs/index.html` is already set up to
show them off.

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/completions` | POST | Routed chat completion. Caller doesn't choose the model. |
| `/v1/models` | GET | Registry of available models and their pricing. |
| `/v1/stats` | GET | Cost savings summary (the headline number). |
| `/v1/routing-config` | GET/PUT | Read or update the tier → model map without redeploying. |
| `/health` | GET | Liveness check. |

### `POST /v1/completions`

Request:

```json
{
  "messages": [
    { "role": "user", "content": "Write a short story about a robot learning to paint" }
  ],
  "max_latency_seconds": null
}
```

`max_latency_seconds` is optional — if the tier's assigned model's average
latency exceeds it, the router reassigns to the fastest model that still
meets the required quality tier (see `app/router/router.py`).

Response:

```json
{
  "text": "In a small, sunlit corner of Vienna...",
  "routed_model": "claude-sonnet",
  "provider": "anthropic",
  "complexity_tier": 3,
  "classifier_confidence": 0.95,
  "input_tokens": 17,
  "output_tokens": 596,
  "cost_usd": 0.01413,
  "latency_seconds": 4.2,
  "request_id": 42,
  "reassigned_for_latency": false
}
```

### `PUT /v1/routing-config`

```json
{
  "tier_to_model": { "1": "llama-local", "2": "gpt-4o-mini", "3": "claude-sonnet" },
  "min_confidence": 0.55,
  "judge_model": "gpt-4o"
}
```

Takes effect immediately, no redeploy — the router's in-memory config cache
is refreshed and the new mapping is persisted back to `config/routing.yaml`.

## Configuration

Environment variables (`.env`, see `.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Required for gpt-4o / gpt-4o-mini calls, the `llm` classifier backend, and the verifier's judge model | — |
| `ANTHROPIC_API_KEY` | Required for claude-sonnet / claude-haiku calls | — |
| `OLLAMA_HOST` | Local Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Local model name (must match a pulled Ollama model) | `llama3.1:8b` |
| `DATABASE_PATH` | SQLite file location | `./data/autopilot.db` |
| `LOG_LEVEL` | Python logging level for structured JSON logs | `INFO` |
| `VERIFIER_JUDGE_MODEL` | Model used for LLM-as-judge scoring and escalations | `gpt-4o` |
| `QUALITY_THRESHOLD` | Verifier score (1-5) below which a response is flagged as a routing failure | `4.0` |
| `AUTO_ESCALATE` | Whether a flagged failure triggers an automatic re-run on the judge model | `true` |

`config/routing.yaml` (hot-reloadable via `PUT /v1/routing-config`):

| Key | Purpose |
|---|---|
| `classifier_backend` | `llm` (default) or `tfidf` — see [The classifier](#the-classifier-two-backends-one-honest-experiment) |
| `tier_to_model` | Maps complexity tier 1/2/3 to a registry key in `app/models/registry.py` |
| `min_confidence` | Below this, escalate one tier before the first request is even sent |
| `judge_model` | Model used by the async verifier |

## Testing & CI

```bash
pytest
```

20 tests, all mocked/pure-function — no API keys or billing required:

| File | Covers |
|---|---|
| `test_registry.py` | `ModelConfig` cost math, registry lookups |
| `test_retry.py` | Retry/backoff recovers from transient errors, gives up and reraises after max attempts, doesn't swallow unrelated exceptions |
| `test_router.py` | Escalation on low classifier confidence, tier-3 escalation ceiling, latency-aware reassignment (and the case where no faster option meets budget) |
| `test_verifier.py` | Score parsing, quality-threshold logic, and the verifier worker's escalation/correct-tier decisions with a mocked judge call |

`pytest.ini` scopes discovery to `tests/` so `pytest` never tries to import
`scripts/test_providers.py` or `scripts/load_test.py` as test modules (their
names otherwise match pytest's default `test_*.py` pattern, and importing
them would either require unrelated dependencies or make real, billed API
calls as a side effect of running the test suite).

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR to `main`:
installs dependencies, fetches the real Dolly-15k dataset, trains the
`tfidf` classifier fallback, and runs the full test suite — all without
needing API keys or billing, so CI stays free and deterministic.

## Observability

Every request produces two things: a row in SQLite (queried by the
dashboard and `/v1/stats`) and a structured JSON log line on stdout (for
Grafana/Loki/CloudWatch-style aggregation) — see `app/logging_config.py`.

## The feedback loop

Every request the verifier flags as a routing failure (cheap model's answer
scored below `QUALITY_THRESHOLD` against the judge model) is stored with a
`correct_tier`. Run `python -m scripts.retrain_from_feedback` on a schedule
(cron / Task Scheduler) to fold those into the training set and retrain —
this is what makes the `tfidf` fallback classifier get better with real
traffic instead of staying frozen at its initial accuracy.

## Known limitations & roadmap

Documented honestly rather than glossed over:

- **In-process verifier worker.** The async verification queue lives in the
  API process's memory — restarting the API drops any in-flight
  verification jobs. A real production deployment would move this to
  Redis/Celery or a managed queue; intentionally out of scope here (see
  `app/verifier/worker.py`).
- **`tfidf` backend ceiling.** ~63-65% accuracy on real, diverse prompts —
  documented in detail above, and the reason the `llm` backend exists.
- **Classifier's own cost/latency tax.** The `llm` backend adds one small
  API call to every request just to decide routing. Cheap in absolute
  terms, but not free — worth measuring against your own traffic mix.
- **No auth or rate limiting** on the FastAPI service — fine for local/demo
  use, not for exposing publicly as-is.
- **Single-turn only.** `messages` accepts a full conversation shape for
  API compatibility, but only the last user message is used for
  routing/classification — no multi-turn context is passed to the
  underlying model.

## License

[MIT](LICENSE)
