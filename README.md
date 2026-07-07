# LLM Cost Autopilot

A routing layer that sits in front of OpenAI, Anthropic, and local Ollama
models, scores each incoming prompt's complexity, routes it to the cheapest
model that can handle it, and continuously verifies that the routing
decision was actually correct — escalating and self-correcting when it
wasn't.

> **Headline metric:** run `scripts/load_test.py` against your own traffic
> and pull the number from `GET /v1/stats` (`savings_pct`). That's the
> number that goes at the top of the portfolio writeup — this README
> doesn't hardcode one because it depends on your prompt mix.

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
                 │  (sklearn, feature-based)     │
                 └──────────────┬───────────────┘
                                 │ tier 1/2/3
                     2. route    ▼
                 ┌─────────────────────────────┐
                 │  Router (config/routing.yaml)│  app/router/
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
  models/       Phase 1 — ModelConfig registry, provider adapters, send_request()
  classifier/   Phase 2 — labeled dataset, TF-IDF + logistic regression training
  router/       Phase 2 — tier -> model mapping (config/routing.yaml), latency-aware reassignment
  verifier/     Phase 3 — LLM-as-judge scoring, async worker, auto-escalation
  db/           Phase 4 — SQLite schema + logging/query helpers
  dashboard/    Phase 4 — Streamlit cost/quality dashboard
  api/          Phase 5 — FastAPI routes + request/response schemas
  logging_config.py  structured JSON request logs (alongside SQLite)
  main.py       Phase 5 — FastAPI app, lifespan (DB init + worker startup)
scripts/
  fetch_real_dataset.py     primary labeled dataset: real prompts from Dolly-15k (see below)
  generate_dataset.py       template-generated fallback/bootstrap dataset (Phase 2)
  test_providers.py         Phase 1 baseline: same prompts across every model
  retrain_from_feedback.py  Phase 3.4 flywheel: failures -> retrain
  load_test.py              Phase 6 load test against a running API
config/routing.yaml         tier -> model map, editable via PUT /v1/routing-config
tests/                       pytest unit tests (testpaths scoped via pytest.ini so
                              `pytest` never tries to collect scripts/*.py)
```

## The classifier's training data — real, not synthetic

`app/classifier/data/labeled_prompts.csv` is 900 real, human-written prompts
sampled from [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
(CC BY-SA 3.0), not templates. `scripts/fetch_real_dataset.py` maps Dolly's
task categories onto our 3 tiers as a documented heuristic:

| Tier | Dolly categories |
|---|---|
| 1 (simple) | open_qa, general_qa, closed_qa, information_extraction |
| 2 (moderate) | classification, summarization |
| 3 (complex) | creative_writing, brainstorming |

**Honest result:** 63.9% held-out accuracy (180-example test set) — well
below the template-generated version's 97.6%, and below the original
spec's "~80% is fine for V1" bar. That gap is the real finding: keyword/
TF-IDF features on real, messy prompts have a ceiling, and template
accuracy numbers are misleadingly easy. The confusion matrix shows tier 2/3
(classification, summarization, creative writing) classify confidently
(72-75% precision); tier 1 is the weak spot, because Dolly's "open_qa"
category lumps genuinely trivial and genuinely nuanced questions under one
label — a real, explainable data-quality limitation, not a training bug.
Getting meaningfully higher would mean an LLM-based classifier (few-shot
with a real model) instead of hand-built features — a good next step once
cloud billing is available, but out of scope for this local, free pass.

Run `python -m scripts.fetch_real_dataset --sample-check 15` to print
random samples and spot-check the category->tier mapping yourself.

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

### 2. Fetch the real dataset and train the classifier (Phase 2)

```bash
python -m scripts.fetch_real_dataset
python -m app.classifier.train
```

This downloads real, human-written prompts (see "The classifier's training
data" above) and trains on them — no API keys needed. If you'd rather
bootstrap quickly with the older template-generated set instead (faster,
but inflated accuracy that doesn't reflect real traffic), use
`python -m scripts.generate_dataset` instead. Either way, the Phase 3
feedback loop appends real verifier-caught routing failures to whichever
dataset you're using.

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

### 5. Load test + generate the portfolio numbers (Phase 6)

```bash
python -m scripts.load_test --n 500 --url http://localhost:8000
curl http://localhost:8000/v1/stats
```

Screenshot the Streamlit dashboard and the `/v1/stats` payload — those are
the artifacts for the writeup.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/completions` | POST | Routed chat completion. Caller doesn't choose the model. Optional `max_latency_seconds` — if the tier's assigned model is too slow, reassigns to the fastest model that still meets the required quality tier. |
| `/v1/models` | GET | Registry of available models and their pricing. |
| `/v1/stats` | GET | Cost savings summary (the headline number). |
| `/v1/routing-config` | GET/PUT | Read or update the tier -> model map without redeploying. |

## Observability

Every request produces two things: a row in SQLite (queried by the
dashboard and `/v1/stats`) and a structured JSON log line on stdout (for
Grafana/Loki/CloudWatch-style aggregation) — see `app/logging_config.py`.

CI (`.github/workflows/ci.yml`) generates the seed dataset, trains the
classifier, and runs `pytest` on every push/PR to `main`.

## The feedback loop

Every request the verifier flags as a routing failure (cheap model's answer
scored below `QUALITY_THRESHOLD` against the judge model) is stored with a
`correct_tier`. Run `python -m scripts.retrain_from_feedback` on a schedule
(cron / Task Scheduler) to fold those into the training set and retrain —
this is what makes the router get better with real traffic instead of
staying frozen at V1 accuracy.
