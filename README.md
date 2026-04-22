# LLM Optimization Gateway

> **Production-grade middleware gateway in front of one or more LLM providers.**
> Optimizes prompt tokens, caches responses (exact + semantic), routes with
> failover, enforces governance, and exposes Prometheus metrics — with a
> polished operations console.

---

## Table of Contents

1. [What this is](#what-this-is)
2. [Architecture overview](#architecture-overview)
3. [Request lifecycle (step-by-step)](#request-lifecycle-step-by-step)
4. [Every module, explained](#every-module-explained)
5. [Local setup](#local-setup)
6. [Running on EC2](#running-on-ec2)
7. [Configuration reference](#configuration-reference)
8. [API reference](#api-reference)
9. [Operations console (UI)](#operations-console-ui)
10. [Observability](#observability)
11. [Testing](#testing)
12. [Packaging and distribution](#packaging-and-distribution)
13. [Enterprise hardening roadmap](#enterprise-hardening-roadmap)
14. [Known limitations](#known-limitations)

---

## What this is

A single binary-free Python service that sits between your applications and
any LLM provider (OpenAI to begin with; the provider interface is pluggable).
Every completion request flows through a deterministic pipeline that
(a) rewrites the prompt more tersely without losing meaning,
(b) checks an exact-match cache and then a semantic-similarity cache,
(c) only then calls the upstream provider,
(d) stores the response back, and
(e) emits metrics and structured logs for every step.

The result: fewer tokens shipped upstream, lower latency on repeats, one
observable chokepoint for all LLM traffic, and provider-agnostic failover.

Two deployment modes are supported out of the box:
- **Internal shared platform** — runs behind a load balancer, uses Redis,
  scraped by Prometheus, fronted by your SSO/gateway.
- **Standalone developer tool** — runs on a laptop or EC2 box with in-process
  cache and no external services. No containers, no Node.

---

## Architecture overview

```
                    ┌─────────────────────────────────────┐
                    │      Client / SDK / Dashboard       │
                    └───────────────────┬─────────────────┘
                                        │ HTTP/JSON
                    ┌───────────────────▼─────────────────┐
                    │         FastAPI  API Layer          │
                    │  /v1/complete  /health  /metrics    │
                    │  /stats  /  (dashboard)             │
                    └───────────────────┬─────────────────┘
                                        │
                    ┌───────────────────▼─────────────────┐
                    │       Gateway Orchestrator          │
                    │  (core/orchestrator.py)             │
                    │  governance → score+coach →         │
                    │  optimize → cache → route →         │
                    │  store → metrics                    │
                    └──┬────────┬────────┬────────────────┘
                       │        │        │
        ┌──────────────▼─┐  ┌───▼───┐  ┌─▼──────────────┐
        │ Coaching       │  │ Opt.  │  │ Routing +      │
        │ - 7-dim score  │  │ clean │  │ Provider       │
        │ - canonical MD │  │ norm. │  │ Adapters       │
        │ - never invent │  │ dedup │  │ echo / openai  │
        │ - suggestions  │  │ compr │  │ failover       │
        │                │  │ prune │  │                │
        └────────────────┘  └───────┘  └────────────────┘
                       │        │        │
                       └────────▼────────┘
                          Caching (exact + semantic)
                          Observability (structlog + prometheus)
```

**Why this layering?** Each concern is an interface, not a helper function,
so the team can swap:
- the cache backend (`memory` ↔ `redis`) with one env var,
- the embedder (`hashing` ↔ `openai`) with one env var,
- the provider chain (comma-separated env var),
without touching the request pipeline. That's the point of a gateway.

---

## Request lifecycle (step-by-step)

A single `POST /v1/complete` traces through these stages. Every stage emits
metrics and structured logs.

1. **API layer** (`api/routes.py`) — validates the JSON body via Pydantic,
   calls `Gateway.complete(request)`. Maps `GatewayError` subclasses to
   HTTP status codes (400/413/502/503/504).

2. **Governance** (`governance/policy.py`) — rejects empty prompts,
   estimates tokens via tiktoken (or char-heuristic fallback), and enforces
   `max_prompt_tokens`.

3. **Prompt Coaching** (`coaching/`) — scores the prompt on 7 dimensions.
   If the composite score is below `LLMGW_PROMPT_QUALITY_THRESHOLD`
   (default 0.55), restructures it into canonical markdown sections.
   **Never invents content** — only reorganizes what the user wrote.
   Emits reasoning strings for every section it creates. See the
   [Prompt Coaching](#prompt-coaching) section below for the full rubric.

4. **Optimization** (`optimization/optimizer.py`) — unless bypassed, runs:
   - `cleaner.py` — minify embedded JSON, strip HTML comments, collapse
     blank-line runs. Particularly effective on RAG prompts.
   - `normalizer.py` — whitespace/Unicode cleanup.
   - `deduper.py` — remove repeated lines/sentences.
   - `compressor.py` — remove filler phrases ("could you please",
     "in order to", "basically"). Respects per-sentence constraint
     detection and code fences.
   - `pruner.py` — entropy-based stopword removal in safe contexts.
     Never touches code, JSON, quoted strings, or constraint sentences.
     This is where the real token savings come from on top of filler
     removal.
   - `guardrails.py` — scores confidence, forces fallback if compression
     invalidated the prompt.

5. **Cache lookup** (`caching/manager.py`) — exact lookup on
   `sha256(model || prompt)`. On miss, if semantic cache is enabled,
   embed the prompt and scan semantic entries for the best cosine
   similarity above threshold.

6. **Routing** (`routing/router.py`) — on cache miss, call the primary
   provider. On `ProviderError` or `ProviderTimeoutError`, failover.

7. **Store** — on upstream success, write the response to both caches
   under the *optimized* prompt, unless `unsafe_to_cache=true`.

8. **Metadata** — the response contains the original, coached, and
   optimized prompts, full prompt-quality report (7 dimensions + reasoning +
   suggestions), token stats, latency, cache status, routing path,
   provider/model used, optimization techniques applied, and warnings.

---

## Prompt Coaching

Just stripping filler words ("please kindly") saves a few tokens per
request but doesn't improve *output quality*. The Coaching layer goes
further: it scores every prompt against seven prompt-engineering
dimensions used by the OpenAI, Anthropic, and PromptBuilder 2025
best-practice frameworks, and rewrites weak prompts into a canonical
markdown structure.

### The 7-dimension rubric

| Dimension | Weight | What it checks |
|---|---|---|
| **task** | 0.25 | Clear action verb (summarize, classify, generate…) with concrete object; not vague ("help me with this"). |
| **format** | 0.18 | Output shape specified — "in JSON", "as 3 bullets", "under 200 words", "return only…". |
| **context** | 0.14 | Background / input material provided or delimited; fenced code blocks count. |
| **specificity** | 0.13 | Measurable criteria — numbers, examples, concrete success definitions. |
| **examples** | 0.10 | Few-shot demonstrations present (`e.g.`, `example:`, `input→output` pairs). |
| **constraints** | 0.10 | Explicit do/don't, must/must-not, only/avoid/exclude boundaries. |
| **role** | 0.10 | Persona established ("You are a…", "Act as…"). |

The composite score (0.0–1.0) is a weighted average, with a penalty
multiplier for sub-4-word prompts (they cannot meaningfully score high
regardless of content).

### What the Coach does

When the score is **below threshold** (default 0.55), the Coach
restructures the prompt into canonical markdown sections:

```
# Task
Summarize the Q4 revenue report in exactly 3 bullet points.

# Context
Here is the text.
<input>
```
Revenue grew 12% to $5.2M driven by enterprise deals.
```
</input>

# Constraints
- Do not include any preamble.
- Only mention revenue figures.

# Success Criteria
Exactly 3 bullets, under 80 words.
```

Every pasted data block is wrapped in `<input>...</input>` — the
delimiter both the OpenAI GPT-4.1 guide and Anthropic's prompt
engineering docs recommend as the most reliable for modern models.

### What the Coach never does

The coach follows a strict rule: **never invent content the user did not
provide**. It will not:

- Fabricate a role ("You are a senior data scientist"). It will
  *suggest* adding one.
- Fabricate examples. It will suggest adding few-shot demonstrations.
- Invent a format if none was stated. It will suggest specifying one.
- Add a fictional persona or tone.

Instead, for every missing element the Coach emits a structured
`suggestions[]` list in the response metadata so the user can see
exactly what was skipped and why.

### Response metadata example

```json
{
  "prompt_quality": {
    "score": 0.42,
    "threshold": 0.55,
    "dimensions": [
      { "name": "role",        "score": 0.0, "weight": 0.10, "reason": "No role or persona specified." },
      { "name": "task",        "score": 1.0, "weight": 0.25, "reason": "Clear task with action verb." },
      { "name": "context",     "score": 1.0, "weight": 0.14, "reason": "Context or input is provided." },
      { "name": "format",      "score": 0.0, "weight": 0.18, "reason": "Output format is not specified." },
      { "name": "constraints", "score": 1.0, "weight": 0.10, "reason": "Multiple constraints are stated." },
      { "name": "examples",    "score": 0.15, "weight": 0.10, "reason": "No examples provided. Few-shot examples dramatically improve output consistency for structured tasks." },
      { "name": "specificity", "score": 1.0, "weight": 0.13, "reason": "Prompt includes concrete, measurable criteria." }
    ],
    "coached": true,
    "techniques_applied": ["structure:task", "structure:context", "structure:constraints"],
    "reasoning": [
      "Front-loaded the action-verb instruction into a `# Task` section — models attend most reliably when the ask appears first.",
      "Grouped background material under `# Context` and wrapped pasted inputs in `<input>` tags so the model can distinguish data from instructions.",
      "Collected negations (do-not / never / must-not / only) under a `# Constraints` section."
    ],
    "suggestions": [
      "Consider adding a `# Role` section (e.g. 'You are an experienced technical writer'). The coach never invents a role on your behalf.",
      "Consider adding 1–3 input→output examples. Few-shot demonstrations dramatically improve consistency.",
      "Specify an output format in a `# Format` section (JSON / bullets / word limit / schema)."
    ]
  }
}
```

### Interaction with optimization

**Coaching runs before optimization.** This is deliberate — the Coach
adds structural scaffolding (section headers, XML tags) that makes the
prompt *longer* but more reliable. The Optimizer then trims any filler
inside the restructured prompt. The net token change depends on the
source prompt: very weak prompts often end up similar in token count
but dramatically stronger; verbose prompts see both structural
improvement *and* token savings.

### Bypassing

Three independent flags control each layer:
- `bypass_coaching` — score but never rewrite (still get the quality report)
- `bypass_optimization` — skip cleaner/normalizer/compressor/pruner
- `bypass_cache` — always hit upstream

---

## Every module, explained

Here's what every file does and why it's there.

### `src/llm_gateway/config/settings.py`
Pydantic-settings class loaded from env vars (`LLMGW_` prefix) or `.env`.
All knobs live here: cache TTL, similarity threshold, provider chain,
OpenAI base URL, Redis URL. No secrets are hardcoded — the OpenAI key
comes from `LLMGW_OPENAI_API_KEY`. `get_settings()` is LRU-cached so the
values are computed once per process.

### `src/llm_gateway/core/models.py`
All Pydantic DTOs. `CompletionRequest` (what clients send),
`CompletionResponse` + `CompletionMetadata` (what we return),
`ProviderRequest`/`ProviderResponse` (internal provider contract), plus
enums for `CacheStatus` (exact_hit/semantic_hit/miss/bypass) and
`RoutingPath` (cache/upstream/failover).

### `src/llm_gateway/core/exceptions.py`
Exception hierarchy rooted at `GatewayError`, each with a `status_code`
and a stable `code` string. The API layer maps these to HTTP responses
without leaking internals.

### `src/llm_gateway/core/orchestrator.py`
The heart of the system. `Gateway.complete(req)` runs the full pipeline
described above. Emits a per-request UUID and binds it to the log context
so every log line for that request is correlated.

### `src/llm_gateway/governance/tokens.py`
`TiktokenEstimator` (accurate, cl100k_base) with `HeuristicEstimator`
fallback. The gateway never crashes because tiktoken isn't available.

### `src/llm_gateway/governance/policy.py`
The policy object called before optimization. Rejects empties, measures
tokens, enforces size limits.

### `src/llm_gateway/optimization/normalizer.py`
Idempotent structural cleanup. Unicode NFC, smart-quote folding, zero-
width removal, whitespace collapse. Safe-by-construction — never changes
meaning.

### `src/llm_gateway/optimization/deduper.py`
Line-level and sentence-level deduplication. Uses a canonical key
(lowercase alnum) for comparison but keeps the original wording of the
first occurrence. Leaves short lines alone so list items and bullets
stay intact.

### `src/llm_gateway/optimization/compressor.py`
Regex-based filler removal. **Key design choice**: before applying
filler patterns, every line is checked against a constraint regex
(`must`, `do not`, `never`, `required`, `format:`, `schema:`, etc.).
Constraint lines are passed through untouched. Content inside fenced
code blocks (\`\`\`…\`\`\`) is never touched either — we track fence
state while scanning lines.

### `src/llm_gateway/optimization/guardrails.py`
Runs after compression. Counts intent-bearing signals in the original
vs optimized text; mismatches penalize confidence. Penalizes aggressive
compression (>70% reduction). Returns `confidence ∈ [0, 1]` plus a list
of warnings. Confidence zero forces the orchestrator to fall back to the
normalized (but uncompressed) prompt.

### `src/llm_gateway/optimization/optimizer.py`
Composes the four stages into a single `optimize()` call and emits an
`OptimizationReport` describing exactly what was applied.

### `src/llm_gateway/caching/base.py`
The `CacheBackend` ABC: async `get/set/delete/clear/size/iter_entries`.
`iter_entries()` is the one that enables semantic scan.

### `src/llm_gateway/caching/memory.py`
In-process `OrderedDict`-backed LRU with TTL, thread-safe via
`asyncio.Lock`. Default backend — zero ops.

### `src/llm_gateway/caching/redis_backend.py`
Drop-in Redis implementation using `redis.asyncio`. Namespaced by
`LLMGW_` prefix so it never flushes anyone else's keys. Raises
`CacheBackendError` on IO failure, which the manager catches and
degrades to a miss.

### `src/llm_gateway/caching/embeddings.py`
Two embedders behind a single `Embedder` protocol:
- `HashingEmbedder` — feature-hashing over character n-grams (3-5) plus
  unigram tokens. Produces deterministic L2-normalized vectors.
  Zero dependencies, no network. This is what lets semantic caching
  work in CI and on offline dev machines.
- `OpenAIEmbedder` — calls `/embeddings` for higher semantic quality.

`cosine_similarity` handles edge cases (empty, zero-norm) defensively.

### `src/llm_gateway/caching/manager.py`
Fronts the backend with exact-then-semantic lookup logic. All cache
errors are caught and logged; the request continues as a miss.

### `src/llm_gateway/providers/base.py`
`LLMProvider` ABC with one method: `complete(ProviderRequest) -> ProviderResponse`.

### `src/llm_gateway/providers/echo.py`
Deterministic offline stub. Lets the gateway boot and pass tests with no
API key.

### `src/llm_gateway/providers/openai_provider.py`
Real adapter for OpenAI's Chat Completions endpoint. Distinguishes
timeouts (→ 504) from other HTTP errors (→ 502), parses `usage` when
present, and closes its httpx client on shutdown.

### `src/llm_gateway/providers/registry.py`
Factory that builds the ordered provider list from `LLMGW_PROVIDER_CHAIN`.
Unknown names raise `ConfigurationError` at startup, not at request time.

### `src/llm_gateway/routing/router.py`
Iterates the provider chain. First provider → `RoutingPath.UPSTREAM`;
subsequent providers → `RoutingPath.FAILOVER`. Records per-provider
latency and failure counts.

### `src/llm_gateway/observability/logging.py`
structlog configured to emit JSON with ISO timestamps. Request context
(`request_id`, etc.) is bound via `contextvars` so concurrent requests
don't mix their log lines.

### `src/llm_gateway/observability/metrics.py`
Prometheus `CollectorRegistry` with counters (requests, errors,
cache hits, provider failures, tokens saved, optimization bypasses),
histograms (end-to-end latency, per-provider upstream latency), and a
gauge (cache entries). Using a dedicated registry keeps tests isolated.

### `src/llm_gateway/api/routes.py`
Endpoints + `RecentRequestsBuffer` ring for the dashboard.

### `src/llm_gateway/api/dependencies.py`
FastAPI `Depends` wiring to `app.state` singletons.

### `src/llm_gateway/main.py`
Application factory. Builds the singletons (metrics → estimator →
governance → optimizer → cache backend + embedder → cache manager →
provider chain → router → gateway), constructs the FastAPI app, and
exposes `app` for `uvicorn` and a `main()` for `python -m llm_gateway.main`.

### `ui/index.html`
Single-file dashboard. Dark editorial theme (IBM Plex Serif + JetBrains
Mono). Left panel: interactive prompt console with before/after view
and KPI tags. Right panel: live telemetry (KPI cards, cache-rate bar,
recent-requests table), auto-refreshed every 3s from `/stats`. Zero
JS dependencies, zero build step.

### `tests/`
- `test_optimizer.py` — unit tests for every optimizer component.
- `test_cache.py` — memory backend, embeddings, layered manager.
- `test_governance.py` — empty and oversize prompt policy.
- `test_api.py` — all endpoints via `httpx.ASGITransport` (no real network).
- `test_e2e.py` — full-pipeline tests including failover, cache bypass,
  unsafe-to-cache, and optimization bypass.

---

## Local setup

### macOS / Linux

```bash
bash scripts/run_local.sh
```

That's it. The script creates a `.venv`, installs deps, copies
`.env.example` → `.env` if missing, and starts the server on
`http://localhost:8080`.

Open the dashboard at **http://localhost:8080**.

### Windows

```bat
scripts\run_local.bat
```

### Manual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m llm_gateway.main
```

### Quick sanity call

```bash
curl -s http://localhost:8080/v1/complete \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Could you please kindly summarize this document for me."}' \
  | python -m json.tool
```

You should see an optimized prompt that drops `please kindly`, non-zero
`tokens_saved`, and `cache_status: miss`. Run the same call again — the
second one comes back as `exact_hit` in single-digit milliseconds.

---

## Running on EC2

This is intentionally boring — no containers required.

```bash
# On a fresh Amazon Linux 2023 / Ubuntu 22.04 box:
sudo dnf install -y python3.11 python3.11-pip git   # or: apt-get install -y python3.11 python3.11-venv

# Copy the zip up (from your laptop), then:
unzip llm-optimization-gateway-1.0.0.zip
cd llm-optimization-gateway

# One-shot run:
bash scripts/run_local.sh

# Persistent run (systemd):
sudo tee /etc/systemd/system/llm-gateway.service > /dev/null <<'EOF'
[Unit]
Description=LLM Optimization Gateway
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/llm-optimization-gateway
EnvironmentFile=/home/ec2-user/llm-optimization-gateway/.env
ExecStart=/home/ec2-user/llm-optimization-gateway/.venv/bin/python -m llm_gateway.main
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now llm-gateway
sudo systemctl status llm-gateway
```

Open port 8080 in the EC2 security group (or front it with an ALB / Nginx).

---

## Configuration reference

All via env vars (or `.env`). Prefix: `LLMGW_`.

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8080` | Bind port |
| `LOG_LEVEL` | `INFO` | structlog level |
| `ENVIRONMENT` | `development` | Label for logs/dashboard |
| `MAX_PROMPT_TOKENS` | `8000` | Governance cap |
| `REJECT_OVERSIZE` | `false` | Reject vs compress oversized prompts |
| `OPTIMIZATION_ENABLED` | `true` | Global kill-switch |
| `AGGRESSIVE_PRUNING` | `true` | Entropy-based stopword removal (on top of filler) |
| `COMPRESSION_CONFIDENCE_WARN_THRESHOLD` | `0.6` | Warn if confidence below this |
| `COACHING_ENABLED` | `true` | Score every prompt + rewrite weak ones |
| `PROMPT_QUALITY_THRESHOLD` | `0.55` | Rewrite prompts scoring below this |
| `CACHE_BACKEND` | `memory` | `memory` or `redis` |
| `CACHE_TTL_SECONDS` | `3600` | Entry TTL |
| `CACHE_MAX_ENTRIES` | `10000` | Memory backend cap |
| `SEMANTIC_CACHE_ENABLED` | `true` | Turn off semantic layer |
| `SEMANTIC_SIMILARITY_THRESHOLD` | `0.92` | Cosine sim cutoff |
| `EMBEDDING_BACKEND` | `hashing` | `hashing` or `openai` |
| `EMBEDDING_DIM` | `256` | Hashing embedder dim |
| `PROVIDER_CHAIN` | `echo` | Comma-separated: `openai,echo` |
| `DEFAULT_MODEL` | `echo-1` | Default model name |
| `OPENAI_API_KEY` | *(unset)* | Required if openai in chain |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | For proxies |
| `OPENAI_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `REDIS_URL` | `redis://localhost:6379/0` | Used when backend=redis |

---

## API reference

### `POST /v1/complete`

**Request**:

```json
{
  "prompt": "Could you please summarize this.",
  "model": "gpt-4o-mini",
  "max_tokens": 256,
  "temperature": 0.2,
  "bypass_optimization": false,
  "bypass_cache": false,
  "unsafe_to_cache": false,
  "tags": {"team": "growth"}
}
```

**Response** (200):

```json
{
  "completion": "…",
  "metadata": {
    "original_prompt": "Could you please summarize this.",
    "optimized_prompt": "summarize this.",
    "tokens": {
      "original_prompt_tokens": 7,
      "optimized_prompt_tokens": 3,
      "completion_tokens": 42,
      "tokens_saved": 4,
      "savings_ratio": 0.5714
    },
    "latency_ms": 312.8,
    "cache_status": "miss",
    "routing_path": "upstream",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "optimization": {
      "applied": true,
      "confidence": 1.0,
      "techniques": ["compress(1)"],
      "warnings": []
    },
    "warnings": [],
    "request_id": "b3c9…"
  }
}
```

**Error codes**:
- 400 `empty_prompt`
- 413 `oversize_prompt`
- 502 `provider_error`
- 503 `all_providers_failed`
- 504 `provider_timeout`

### `GET /health`
Liveness + config summary. Useful as an ALB target-group health check.

### `GET /metrics`
Prometheus exposition format. Scrape interval 15-30s is fine.

### `GET /stats`
JSON snapshot used by the dashboard (totals, cache breakdown, recent requests).

### `GET /`
The operations console HTML.

---

## Operations console (UI)

Open `http://localhost:8080/`.

- **Completion Console** (left) — invoke the gateway interactively. Before/
  after prompt views with token counts; latency, savings, cache status,
  routing path, provider, and confidence shown as chips. Warnings render
  in a separate block when optimization confidence is low or intent
  signals shifted.
- **Gateway Telemetry** (right) — tokens saved, total requests, average
  latency, cache entry count, cache-rate distribution, and recent-requests
  table. Auto-refreshes every 3 seconds.

Three toggles on the console let you reproduce edge cases:
- `Bypass optimization` — send the raw prompt verbatim.
- `Bypass cache` — always hit upstream.
- `Unsafe to cache` — hit upstream and skip the store-back step.

---

## Observability

**Structured logs** — every request emits a set of JSON log lines
correlated by `request_id`. Example:

```json
{"event":"gateway.optimized","request_id":"b3c9…","original_tokens":47,
 "optimized_tokens":22,"saved":25,"confidence":1.0,
 "techniques":["normalize","compress(4)"],"level":"info","timestamp":"…"}
```

**Prometheus metrics** — everything you need for an SLO dashboard:

| Metric | Type | Labels |
|---|---|---|
| `llmgw_requests_total` | counter | — |
| `llmgw_errors_total` | counter | `code` |
| `llmgw_cache_hits_total` | counter | `kind=exact|semantic` |
| `llmgw_cache_misses_total` | counter | — |
| `llmgw_tokens_saved_total` | counter | — |
| `llmgw_provider_failures_total` | counter | `provider` |
| `llmgw_optimization_bypasses_total` | counter | — |
| `llmgw_request_latency_seconds` | histogram | — |
| `llmgw_upstream_latency_seconds` | histogram | `provider` |
| `llmgw_cache_entries` | gauge | `kind` |

A sample alert rule:

```yaml
- alert: LLMGatewayUpstreamFailureRateHigh
  expr: sum(rate(llmgw_provider_failures_total[5m])) / sum(rate(llmgw_requests_total[5m])) > 0.05
  for: 10m
```

---

## Testing

```bash
bash scripts/run_tests.sh
# or: pytest -v
```

Everything runs offline against `EchoProvider` and `HashingEmbedder`.

Categories:
- `tests/test_optimizer.py` — normalize/dedupe/compress/guardrails units.
- `tests/test_cache.py` — memory backend, embeddings, manager.
- `tests/test_governance.py` — empty/oversize prompt policy.
- `tests/test_api.py` — ASGI-level endpoint tests (health, metrics,
  /v1/complete, oversize rejection, dashboard served).
- `tests/test_e2e.py` — end-to-end happy path, exact hit on re-call,
  semantic hit on paraphrase, bypass_cache, bypass_optimization,
  unsafe_to_cache, provider failover, all-providers-failed.

---

## Packaging and distribution

```bash
python package_release.py
```

Produces `dist/llm-optimization-gateway-<version>.zip` containing the
full source, tests, scripts, and UI — but **not** `.venv`, `__pycache__`,
`.env`, or any IDE artifacts. Extract anywhere and run:

```bash
unzip llm-optimization-gateway-1.0.0.zip
cd llm-optimization-gateway
bash scripts/run_local.sh
```

---

## Enterprise hardening roadmap

The current version is production-grade for single-tenant or small-team
use. The next layer of hardening for a shared internal platform:

1. **Authentication** — per-team API keys with rate limits. Plug in an
   auth dependency on `/v1/complete`. The hook point is
   `api/dependencies.py`.
2. **Per-team cost budgets** — extend `governance/` with a `BudgetPolicy`
   that reads a ledger and rejects / degrades when a team exhausts
   quota. Metrics labels already support per-team breakdowns via `tags`.
3. **Cheap-vs-expensive model routing** — the `Router` is ready; add a
   pre-route hook that classifies prompts (length, complexity score)
   and picks between e.g. `gpt-4o-mini` and `gpt-4o`.
4. **Response streaming** — add an async-iterator path to
   `LLMProvider.complete_stream` and pipe it through SSE. Caching
   streaming responses requires buffering on cache-fill only.
5. **Distributed semantic cache** — replace the in-process scan with a
   vector store (pgvector, Qdrant) behind the same `CacheBackend`
   interface; only `iter_entries` needs replacing.
6. **Circuit breaker per provider** — wrap `router.complete` with a
   breaker (e.g. `purgatory`) to short-circuit dead providers for N
   seconds.
7. **Request/response redaction** — pluggable PII scrubber before logs
   are emitted.
8. **Multi-replica coherence** — with Redis backend already supported,
   pair with a shared lock for cache-fill coalescing (thundering-herd
   avoidance on cold-miss bursts).
9. **OpenTelemetry traces** — swap `structlog` binds for OTel spans; the
   orchestrator is already the natural span boundary.
10. **Admin endpoints** — `/v1/cache/invalidate`, `/v1/providers/health`,
    `/v1/governance/limits`, gated by admin auth.

---

## Known limitations

- **Semantic cache search is O(N)** over live entries. Fine up to ~10K
  entries in-process; swap to a vector store past that scale (see
  roadmap item 5).
- **Hashing embedder** is good at catching near-exact paraphrases but
  is not a substitute for a real embedding model for diverse prompts.
  Production deployments should use `LLMGW_EMBEDDING_BACKEND=openai`.
- **Compressor** is regex-based and English-heavy. It deliberately
  under-compresses (never touching constraints or code) — this is the
  right default for a gateway but means token savings on technical
  prompts will be modest. For higher savings, a small LLM-based
  rewriter could be added as a fourth optimization stage.
- **Single-process cache** by default. For multiple replicas, use
  `LLMGW_CACHE_BACKEND=redis`.
- **No streaming** in v1. See roadmap.
- **No auth** in v1. Put it behind an authenticating reverse proxy for
  now, or add one via FastAPI dependencies.
