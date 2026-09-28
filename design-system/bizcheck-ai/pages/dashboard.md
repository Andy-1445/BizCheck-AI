# BizCheck AI — Dashboard override

This page overrides the generic enterprise-gateway pattern in `MASTER.md`.

## Product framing

- Audience: students, freelancers, and small businesses checking a company before cooperation.
- Single job: enter a company name or tax ID and understand the public registration record.
- Interface model: a public-data verification workbench, not a marketing landing page.

## Revised tokens

- Ink: `#10243E` — headings and primary text.
- Registry blue: `#1E40AF` — data labels, links, and focus.
- Verification teal: `#0F766E` — primary action and successful registration state.
- Paper: `#F6F8FB` — application background.
- Rule: `#D8E0EB` — dividers and the verification rail.
- Amber: `#9A6700` — warning only, never the primary CTA.

Use Fira Sans for interface text and Fira Code for tax IDs, status codes, dates,
and small data labels. The signature element is a three-stop verification rail
that connects query, candidate selection, and normalized registration data.

## Layout

```text
┌────────────────────────────────────────────────────┐
│ brand                            public-data scope │
├────────────────────────────────────────────────────┤
│ thesis + search                                   │
│ query ──────────────── [查詢公司]                 │
│  ● 輸入       ○ 選擇公司       ○ 檢視登記資料     │
├────────────────────────────────────────────────────┤
│ candidate list OR normalized company record       │
├────────────────────────────────────────────────────┤
│ source, fetched time, governance note              │
└────────────────────────────────────────────────────┘
```

## Interaction rules

### W4-D16-01 unified dashboard (2026-09-24)

- Retain registry navy `#10243E`, blue `#1E40AF`, verification teal `#0F766E`,
  paper `#F6F8FB`, border `#D8E0EB`, and amber `#9A6700` for warnings.
- Use the existing local Chinese sans-serif stack for headings/body and monospace
  for IDs/numbers; do not introduce a remote font dependency.
- Reading order: company identity → basic facts → BizScore → AI → business scope.
  Four in-document links support returning to each section.
- Shared 1200px content width, 24px report gutters (18px on mobile), 14px report
  corners, flat white surfaces, and matching AI single/PK headers and boundaries.
- Keep the three-step verification rail as the product signature. Remove notebook
  ruling and ornamental AI NOTE tabs; do not turn the workbench into a landing page.
- Narrow layouts stack score/report columns and preserve the existing comparison
  cards. No clipping, hidden warnings, ranking, or changes to scoring logic.
- Reading comfort follow-up (2026-09-24): explanatory body copy, limitations,
  disclaimers, evidence disclosures, stale/partial notices and benchmark captions
  use 14–15px with 1.6–1.75 line height; essential facts stay prominent, while
  tax IDs, snapshot IDs and short axis ticks may remain compact secondary labels.
- Implementation: `frontend/src/dashboard.css`, imported after domain/state styles.

- Submit only; do not query on every keystroke.
- Numeric input must contain exactly eight digits before an API request.
- Every async action shows progress and prevents duplicate submission.
- Candidate rows are full-width buttons with visible keyboard focus.
- Errors explain what happened and provide a retry action when `retryable=true`.
- `partial=true` remains usable but is announced before company data.
- Motion is limited to 180–240 ms opacity/position transitions and disabled by
  `prefers-reduced-motion`.
