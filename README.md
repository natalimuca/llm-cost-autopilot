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

## Project showcase site

`docs/index.html` is a self-contained, single-file case-study page (fonts
embedded, no build step) with the real numbers from the run documented
below and the honest classifier-accuracy experiment log. Open it directly
in a browser, or enable **Settings → Pages → branch `main`, folder `/docs`**
on the GitHub repo to host it for free.

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
| 1 | `open_qa`/`general_qa` (context-free trivia) included in tier 1 | 63.9% |
| 2 | Dropped those two categories — narrower but cleaner labels | **84.4%** |
| 3 | Added them back with real per-example GPT-4o labels instead of a category guess | 64.6% |

Step 2 looked like a win, but step 3 revealed why it wasn't a real fix:
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

These numbers were measured with tier 3 → gpt-4o. `config/routing.yaml`
now maps tier 3 → **claude-sonnet** instead, so the live system is a real
three-provider mix (free local Llama, OpenAI mid-tier, Anthropic top-tier)
rather than OpenAI-only with Anthropic registered but unused.
That's the entire point of the escalation loop working as designed, not a
side effect to explain away.

Run `python -m scripts.fetch_real_dataset --sample-check 15` to spot-check
the Dolly category->tier mapping, or `python -m scripts.label_with_llm --n 400`
to re-run the GPT-4o labeling pass (note: subject to your account's
tokens-per-minute rate limit -- ~285/400 succeeded at concurrency 10 on a
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

### 2. Set up the classifier (Phase 2)

The default `classifier_backend: llm` in `config/routing.yaml` needs no
training step at all — it calls gpt-4o-mini directly. If you want the free
offline `tfidf` fallback available too (e.g. for CI or no-billing dev):

```bash
python -m scripts.fetch_real_dataset
python -m app.classifier.train
```

This downloads real, human-written prompts (see "The classifier" above) and
trains on them. If you'd rather bootstrap quickly with the older
template-generated set instead (faster, but inflated accuracy that doesn't
reflect real traffic), use `python -m scripts.generate_dataset` instead.
Either way, the Phase 3 feedback loop appends real verifier-caught routing
failures to whichever dataset you're using.

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
