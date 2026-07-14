# smc-ai-eng — Financial Q&A Chatbot (Take-Home Assignment)

## What this is

A locally-run Q&A chatbot that answers financial questions about U.S. public
companies, grounded in two data sources:

- **SQL (PostgreSQL)** — structured income-statement financials.
- **Vector DB (Pinecone local)** — chunked/embedded text from four FY2025 10-K filings.

**Hard requirement: no hallucination.** If the data needed to answer a question
isn't available (e.g. no 10-K text for a company, or asking about a year outside
2022-2025), the assistant must say so explicitly rather than invent an answer.

Full spec: `Take-Home Task.pdf` (repo root).

## Provided data (do not regenerate — load as-is)

| Path | Contents |
|---|---|
| `10k_filings/*.pdf` | Raw FY2025 10-Ks: Alphabet, Amazon, Apple, Meta. |
| `data/financial_data.sql` | pg_dump-style file (DDL + `COPY FROM stdin`). Table `financial_data`: `company, ticker, sector, year, revenue, gross_profit, operating_income, net_income` (USD). ~48 companies, years 2022-2025, 192 rows. Some cells are `\N` (NULL) — not every company has every field. |
| `data/pinecone_vectors.jsonl.gz` | Pre-chunked + pre-embedded 10-K text, one JSON object per line: `id, namespace, values` (embedding), `metadata` (chunk text, page, source). Chunked with `chunk_size=1000, overlap=200`, embedded with `text-embedding-3-small` at `dimensions=512`, cosine index. Upsert directly — no need to re-embed unless changing chunk/embedding params. |

**Coverage gap is intentional**: SQL covers ~48 companies; the vector store only
covers Alphabet/Amazon/Apple/Meta. A company (e.g. Microsoft) can have SQL
figures but no 10-K text — the agent must recognize this and say the qualitative
grounding isn't available rather than fabricate it.

## Required tech stack

- **Backend**: FastAPI (Python) preferred (Express.js/NestJS acceptable).
- **Frontend**: unconstrained.
- **LLM/embeddings**: OpenAI (key provided separately, $10 budget cap). Use an
  agent framework — LangChain or LangGraph — to route each question to the
  right retrieval source (SQL vs. vector vs. both) and to refuse when neither
  covers the question.
- **Databases**: Postgres + `pinecone-local` (Pinecone's local Docker image,
  point the SDK at it via host override), both brought up via Docker Compose.
- **Auth**: required, must be clean/extensible — will be built on live during
  a follow-up coding session, so keep the auth and data-access layers decoupled
  and simple to extend.

## Baseline acceptance questions (must answer correctly)

1. Net income summary for Apple, 2022-2025 (SQL only).
2. Compare Google vs. Facebook revenue structure and business strategy, 2025
   (vector/10-K only — qualitative).
3. Which of Microsoft/Apple/Google/Facebook had the highest revenue growth
   2024-2025, and why (SQL for the numbers + vector for the "why" — note
   Microsoft has no 10-K in the vector store, so the "why" for Microsoft
   specifically cannot be grounded).

Questions are asked in Thai in the spec; answers should work in whatever
language the user asks in.

## Deliverables

1. Git repo: app source, `docker-compose.yml` for the local data stack, seed/loader scripts.
2. README: how to bring up the databases, load the data, run the app.
3. Running app that correctly answers the three baseline questions above.

## Repo conventions

- `agent-output/` — scratch space for AI-agent-generated artifacts (analysis
  notes, generated reports, intermediate outputs) during development. Gitignored;
  never a source of truth, never referenced by the app.
- Work happens on `dev`; keep `main` as the clean baseline branch.
