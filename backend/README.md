# BizCheck AI Backend

FastAPI backend skeleton for BizCheck AI.

## Requirements

- Python 3.10+

## Local setup

PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` only when local overrides are needed. The application has safe development defaults and does not require an API key.

## Deployment CORS

Development permits both `localhost` and `127.0.0.1` on ports 5173 and 3000.
For a separate production frontend and API, configure the exact HTTPS frontend
origins before starting the backend:

```text
BIZCHECK_ENVIRONMENT=production
BIZCHECK_CORS_ORIGINS=https://app.example.com,https://www.example.com
```

Production fails closed when the allowlist is empty, uses HTTP, or contains a
wildcard, path, query, credentials, or invalid port. The API allows browser GET
and POST requests without credentials and caches successful preflights for 600
seconds. See [`../docs/deployment-cors-v1.md`](../docs/deployment-cors-v1.md)
for the frontend setting, validation rules, and acceptance evidence.

## Run

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Useful URLs:

- API metadata: `http://127.0.0.1:8000/`
- Liveness: `http://127.0.0.1:8000/health/live`
- Readiness: `http://127.0.0.1:8000/health/ready`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Test

```powershell
pytest
```

Replay the versioned offline AI hallucination and boundary suite through the
production validator and service fail-closed path:

```powershell
.\.venv\Scripts\python.exe scripts\run_ai_safety_case_set.py
.\.venv\Scripts\python.exe -m pytest tests\test_ai_safety_case_set.py
```

This suite does not call a live or paid model. Its machine-readable 38-case
catalog is stored at `../samples/llm/ai-safety-boundary-case-set-v1.json`.

## Current API surface

| Method | Path | Current behavior |
|---|---|---|
| GET | `/health/live` | Confirms the FastAPI process is running |
| GET | `/health/ready` | Checks the backend process, live GCIS reachability, and the complete A-J Benchmark catalog; returns structured `200 ready` or `503 not_ready` |
| GET | `/api/v1/companies/search?q=宏碁&status=01` | Searches GCIS and returns normalized company candidates |
| GET | `/api/v1/companies/{tax_id}` | Merges GCIS A1/A3 into a normalized company response |
| GET | `/api/v1/companies/{tax_id}/bizscore` | Returns the company plus the five-dimension BizScore v1 selected through the formal Benchmark catalog |
| POST | `/api/v1/companies/compare` | Atomically returns request-ordered CompanyData, BizScore, and comparison metrics for 2–3 companies |
| POST | `/api/v1/companies/compare/analysis` | Returns validated AI comparison observations, an exact-input cache result, or an explicit deterministic fallback |
| POST | `/api/v1/companies/{tax_id}/analysis` | Builds verified CompanyData + BizScore input and returns Prompt v1 analysis only after schema and safety validation |

The comparison request body is `{"tax_ids":["20828393","47217677"]}`. It
accepts exactly two or three unique strings made of eight ASCII digits and keeps
the request order; a fatal error for any company fails the whole request without
partial items. The response preserves `null`, numeric zero, source-partial and
provisional-score states. PR values from different industry snapshots describe
each company's position in its own peer group and must not be treated as one
shared ranking. Comparison age is recalculated at the shared Benchmark as-of
date, and flattened metrics are checked against their nested CompanyData and
BizScore sources before release. The response also includes the fixed comparison
disclaimer. See
[`../docs/company-comparison-api-v1.md`](../docs/company-comparison-api-v1.md)
for the complete v1 contract and error semantics.

## Enable the AI company-analysis endpoint

The application starts with AI generation disabled, so Week 1 and Week 2 APIs do
not require an LLM credential. Select exactly one provider when AI analysis is
needed.

OpenAI Responses API:

```powershell
$env:BIZCHECK_LLM_PROVIDER = "openai"
$env:BIZCHECK_OPENAI_API_KEY = "<your-api-key>"
$env:BIZCHECK_OPENAI_MODEL = "gpt-5.4"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Tunghai University Chat Completions API:

```powershell
$env:BIZCHECK_LLM_PROVIDER = "thu"
$env:BIZCHECK_THU_API_KEY = "<your-thu-api-key>"
$env:BIZCHECK_THU_BASE_URL = "https://api.ithu.tw/v1"
$env:BIZCHECK_THU_MODEL = "gpt-oss-120b"
$env:BIZCHECK_THU_JSON_MODE = "auto"
$env:BIZCHECK_THU_TEMPERATURE = "0.1"
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Do not commit a real API key. `OPENAI_API_KEY` is also accepted for the OpenAI
provider when the BizCheck-specific key is absent; it is never reused for THU.
The model name, base URL, 45-second timeout, 3,000-token output limit, and
one-attempt default can all be overridden with the variables documented in
`.env.example`. `BIZCHECK_LLM_MAX_ATTEMPTS` accepts 1–3; the default is 2 so a
schema or safety rejection gets one bounded repair retry. THU `auto` JSON mode probes
`json_schema`, then `json_object`, and falls back to the documented plain request
only when the previous format is rejected with HTTP 400/422. See
[`../docs/thu-llm-provider-v1.md`](../docs/thu-llm-provider-v1.md) for the THU
adapter contract and current verification boundary.

Generate one report without a request body:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/companies/20828393/analysis
```

The response contains the exact verified LLM input, the validated structured
analysis, and audit metadata such as the provider response ID, actual model,
Prompt hash, input/output hashes, latency, attempt count, and token usage. Prompt
text, API credentials, and rejected raw model output are never returned. With no
configured key, this route fails closed as `503 AI_ANALYSIS_NOT_CONFIGURED` while
the existing company and BizScore routes remain available.

### AI PK analysis, result cache, and fallback

Generate an optional AI verification memo for the current two- or three-company
comparison:

```powershell
$body = @{
  tax_ids = @("20828393", "22099131")
  force_refresh = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/companies/compare/analysis `
  -ContentType "application/json" `
  -Body $body
```

The endpoint always rebuilds the atomic deterministic comparison first. A GCIS
or Benchmark failure therefore remains a non-2xx error and never calls the model.
When deterministic comparison succeeds, the endpoint returns HTTP `200` in one
of three explicit modes:

- A newly generated, schema-validated AI memo (`cache.status` is `miss` or
  `bypass`).
- A previously validated result for the exact same substantive input (`hit`), or
  a clearly labelled validated old result when refresh failed during the allowed
  stale window (`stale`).
- A deterministic fallback with `analysis: null` when no safe model result is
  available. Fallbacks, provider errors, and rejected raw model output are never
  cached or exposed as AI content.

`force_refresh: true` bypasses a fresh result, but an exact-input stale result may
still be returned if the provider attempt fails. Cache keys include request
order, substantive comparison data, Schema and Prompt versions, provider, and
model; volatile fetch and response-generation timestamps are excluded. Any
substantive company, BizScore, Benchmark, order, contract, provider, or model
change creates a different key.

AI PK v1 uses a bounded application-process LRU cache. Defaults are 128 entries,
15 minutes fresh TTL, and an additional 60-minute stale-if-error window. Override
them with the three `BIZCHECK_LLM_COMPARISON_CACHE_*` variables in
`.env.example`. This cache resets whenever the API process restarts and is not
shared across multiple Uvicorn workers; production scale-out should replace it
with a shared cache while preserving the same fingerprint and validation rules.

The AI PK output only supplies evidence-linked neutral observations and
cooperation-before-verification questions. It cannot recompute BizScore, rank the
companies, name a winner, or recommend a cooperation decision. Cross-industry PR
values remain in their separate peer populations.

The production adapter uses the OpenAI Responses API with `store: false` and
`text.format.type: json_schema`. The provider-specific strict Schema is derived
from (but never mutates) the canonical public JSON Schema; the returned object is
then validated again by Pydantic and the deterministic Prompt v1 guardrails.

The GCIS HTTP client is implemented in `app/services/gcis.py`, and the pure
normalization layer is implemented in `app/services/company_normalizer.py`.
Both are connected to the company routes through an application-scoped client.
Recorded samples under `../samples/gcis/` are used for deterministic tests, so
the test suite does not depend on the live GCIS service.

Normal GCIS requests use a configurable timeout, retry timeouts, connection
errors, and HTTP `429/500/502/503/504`, and expose a typed error category plus
the actual attempt count. `BIZCHECK_GCIS_MAX_RETRIES` means retries after the
first attempt and is limited to `0-3`; exponential delays are capped at 10
seconds. The readiness probe has its own short timeout and deliberately performs
one attempt so health polling does not amplify an upstream outage. See
[`../docs/health-readiness-gcis-resilience-v1.md`](../docs/health-readiness-gcis-resilience-v1.md)
for the response contract and error matrix.

Normalized successful GCIS search and company responses also use a bounded
application-process LRU cache. The defaults are 256 entries, 5 minutes of fresh
TTL, and an additional 6-hour stale-if-error window. After the fresh TTL, only
transient timeout, connection, rate-limit, upstream availability, invalid
response, or normalization failures may use the last successful snapshot;
validation and not-found responses never resurrect cached data. API metadata
reports `live`, `fresh_cache`, or `stale_cache`, preserves the original
`fetched_at`, and requires a `fallback_reason` for stale data. Override the
defaults with `BIZCHECK_GCIS_CACHE_MAX_ENTRIES`,
`BIZCHECK_GCIS_CACHE_TTL_SECONDS`, and
`BIZCHECK_GCIS_CACHE_STALE_IF_ERROR_SECONDS`. The cache resets on process
restart and is not shared across workers. See
[`../docs/gcis-cache-fallback-v1.md`](../docs/gcis-cache-fallback-v1.md) for the
contract, user-facing states, limitations, and acceptance evidence.

The BizScore v1 components are implemented as pure functions in
`app/services/bizscore.py`. They currently cover company age, registration
status, registered-capital scale, registration-change recency, versioned GCIS
industry classification, and peer-percentile scoring, including missing-value
behavior, non-current status handling, total-score caps, cross-date validation,
generic-code exclusion, traceable comparison-group selection, midrank ties, and
the 30-company valid-sample minimum. The total engine applies the 80% coverage
gate, missing-dimension behavior, half-up normalization, status caps, and the four
approved score bands.

Versioned peer Benchmark snapshots use SQLite through
`app/services/benchmark_snapshot.py`. The store atomically filters canonical
company records, recomputes every age against one reference date, records source
and exclusion metadata, prevents version overwrite, and reads one industry group
directly into the peer-percentile scorer. See `../docs/benchmark-snapshot-v1.md`
for the JSONL build command and production data requirements.

The production catalog `benchmark-catalog-2026-08-01-v1` covers nationwide GCIS
categories A-J using ten non-overlapping regional sources per category. The ten
immutable category snapshots contain 1,803,462 eligible membership samples in
total; this is not a unique-company count because one company may belong to more
than one category. Manifests, source checksums, canonical import reports,
snapshot metadata, and the catalog are stored under `data/imports/` and
`data/benchmarks/`.

`GET /api/v1/companies/{tax_id}/bizscore` classifies the company's primary GCIS
business category, selects the category's immutable snapshot through that catalog,
and calculates peer midranks and medians directly in SQLite. The response exposes
`peer_benchmark` as structured data containing the group, snapshot, sample size,
age PR, registered-capital PR, peer index, company-age median, and registered-capital
median. Medians are cached by immutable snapshot and industry after their first
read. Every date-dependent dimension is recomputed against the catalog date
`2026-08-01`, so a later API request does not silently mix a current age with an
older peer snapshot.

The recorded ten-company acceptance set is stored at
`../samples/bizscore/bizscore-v1-10-company-test-set-2026-08-23.json`, with a
human-readable report at `../docs/bizscore-v1-10-company-validation.md`. It covers
three numeric score bands and seven primary industries. Re-run the deterministic
formal-snapshot check with:

```powershell
python scripts/build_bizscore_case_set.py `
  --verify ../samples/bizscore/bizscore-v1-10-company-test-set-2026-08-23.json
```

Run the live Week 2 API acceptance against all ten recorded companies with:

```powershell
python scripts/verify_week2_api.py `
  --case-set ../samples/bizscore/bizscore-v1-10-company-test-set-2026-08-23.json `
  --output ../docs/qa-results/week2-api-acceptance-2026-08-23.json
```
