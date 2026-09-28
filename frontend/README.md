# BizCheck AI Frontend

React + TypeScript + Vite frontend for company lookup, BizScore v1 and the
single-company AI analysis report.

The company view calls `GET /api/v1/companies/{tax_id}/bizscore` and displays:

- company registration facts and business items;
- BizScore total, band, coverage and provisional state;
- a deterministic, non-AI "why this score" explanation for complete,
  provisional and unscored states;
- five score dimensions with plain-language reasons and traceable evidence;
- peer group, sample size, age/capital PR, medians and peer index;
- Benchmark version, data date and non-credit-rating disclaimer.

The company view also exposes an explicit `產生 AI 分析` action. It calls
`POST /api/v1/companies/{tax_id}/analysis` only after the user clicks, then
renders validated observations, verification questions, limitations, the fixed
AI disclaimer and expandable evidence. AI errors remain inside the report
section, so the company facts and BizScore stay usable.

The score explanation never calls the AI endpoint and never recomputes the
total in the browser. The AI section explicitly links back to the fixed-rule
explanation and states that AI cannot add, subtract or replace BizScore points.

Requirements: Node.js 20+ and pnpm.

## Run locally

Start the backend first:

```powershell
cd backend
.\.venv\Scripts\uvicorn.exe app.main:app --reload --host 127.0.0.1 --port 8000
```

Then start the frontend in a second terminal:

```powershell
cd frontend
pnpm install
pnpm dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to the local FastAPI server.

Use `VITE_API_BASE_URL` only when the API is hosted at a different origin.
For example, a separately deployed API can use
`VITE_API_BASE_URL=https://api.example.com/api/v1` at frontend build time. The
backend `BIZCHECK_CORS_ORIGINS` must then contain the exact HTTPS origin shown in
the user's browser for this frontend, such as `https://app.example.com`; it must
not contain the API URL or `*`. See
[`../docs/deployment-cors-v1.md`](../docs/deployment-cors-v1.md).

The backend defaults to a disabled AI provider. In that mode, clicking the
analysis action returns `503 AI_ANALYSIS_NOT_CONFIGURED` without affecting the
other APIs. Configure the backend provider separately before making real model
calls; do not place provider keys in frontend environment variables.

## Week 2 browser acceptance

With the backend and Vite dev server running, execute:

```powershell
node e2e/week2-browser-acceptance.mjs `
  ../docs/qa-screenshots/2026-08-23-week2-final-acceptance
```

The flow verifies all ten formal companies, responsive layouts, provisional and
no-score states, Benchmark failure recovery, and unexpected console/network
errors.
