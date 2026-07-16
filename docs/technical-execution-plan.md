# Technical Execution Plan — Financial Q&A Chatbot

*(Companion to `implementation-plan-v2.md` (strategy, v2.5). This is the code-level detail layer: concrete enough that implementation needs zero design pauses. Execution decisions made here: canonical company keys = **SQL names** (`Google`, `Meta`, `Apple`, `Amazon`); `agent_ro` fixed dev password in plain SQL (initdb `.sql` has no env substitution); verify-veto mechanics = **"render only the last text part"** frontend rule; synthesize streaming = **server-side incremental extraction of the `answer` field** from the strict-JSON token stream (`AnswerFieldExtractor`); intent gate lives inside the route node's structured output (no extra LLM call).)*

## E1. docker-compose.yml

```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: smc-postgres
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app_local_dev        # local-only; never used outside compose
      POSTGRES_DB: findata
    ports:
      - "5432:5432"
    volumes:
      - pgdata_16:/var/lib/postgresql/data    # versioned name — survives image major bumps
      - ./data/financial_data.sql:/docker-entrypoint-initdb.d/01_financial_data.sql:ro
      - ./scripts/initdb/02_roles.sql:/docker-entrypoint-initdb.d/02_roles.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d findata"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 10s

  pinecone:
    image: ghcr.io/pinecone-io/pinecone-local:latest
    container_name: smc-pinecone
    platform: linux/amd64
    environment:
      PORT: "5080"                # control plane
      PINECONE_HOST: localhost    # hosts returned by describe_index → localhost:508x, resolvable from host
    ports:
      - "5080-5090:5080-5090"     # 5080 = control plane; each index gets a data-plane port in 5081-5090
    # no healthcheck: image has no shell/curl; readiness handled by load_pinecone.py retry loop

volumes:
  pgdata_16:
```

- **initdb runs only on first volume creation** → `make down` must be `docker compose down -v` so re-up re-runs `01_` + `02_`. Alphabetical order guarantees table-before-GRANT.
- **No `.sh` initdb scripts** — `.sql` files are immune to Windows CRLF shebang breakage.
- Windows: single-file bind mounts fail cryptically if the host file is missing (Docker creates a directory); `platform: linux/amd64` is a no-op on x86_64 Windows; port-range publish works on Docker Desktop.
- pinecone-local ignores API keys and does **not support named namespaces** — data's `"__default__"` is the default namespace; omit the namespace arg everywhere.

## E2. .env.example + config.py

```bash
# --- OpenAI (required, no default) ---
OPENAI_API_KEY=sk-...                      # provided key ($10 cap); swap to personal key if cap hit
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
EMBED_DIMENSIONS=512                       # must match the pre-embedded vectors
# --- Postgres ---
DATABASE_URL=postgresql+psycopg://app:app_local_dev@localhost:5432/findata
AGENT_RO_DATABASE_URL=postgresql+psycopg://agent_ro:agent_ro_local_dev@localhost:5432/findata
# --- Pinecone local ---
PINECONE_HOST=http://localhost:5080        # control plane; data-plane host resolved via describe_index
PINECONE_INDEX=10k-filings
# --- Auth ---
JWT_SECRET=dev-only-change-me
JWT_EXPIRY_MIN=60
# --- Agent tuning knobs (Day-3 tuning is env-only) ---
SCORE_FLOOR=0.25
TOP_K=6
SQL_ROW_LIMIT=100
# --- Web ---
CORS_ORIGINS=["http://localhost:3000"]
```

`config.py`: `Settings(BaseSettings)` with `SettingsConfigDict(env_file=".env", extra="ignore")`, one typed field per var above (`openai_api_key: str` required — no default; the rest defaulted as in `.env.example`); `get_settings()` lru_cache'd FastAPI dep; tests construct `Settings(openai_api_key="test", ...)` directly. Frontend: `frontend/.env.local` → `NEXT_PUBLIC_API_URL=http://localhost:8000`.

## E3. scripts/initdb/02_roles.sql

```sql
-- agent_ro: the ONLY role the LLM-generated-SQL path connects as.
-- LOCAL DEV ONLY password — this stack never leaves localhost.
CREATE ROLE agent_ro LOGIN PASSWORD 'agent_ro_local_dev';
GRANT CONNECT ON DATABASE findata TO agent_ro;
GRANT USAGE ON SCHEMA public TO agent_ro;
GRANT SELECT ON TABLE financial_data TO agent_ro;
-- Deliberately NO other grants: users table (created later by role "app") is invisible.
ALTER ROLE agent_ro SET default_transaction_read_only = on;
ALTER ROLE agent_ro SET statement_timeout = '5s';
```

Fixed password because initdb executes `.sql` verbatim (no env substitution; only `.sh` sees env, rejected for CRLF reasons). README notes the production path (secret-managed init or migration).

## E4. scripts/load_pinecone.py (stepwise)

REST client (`from pinecone import Pinecone`), not gRPC — avoids the `pinecone[grpc]` extra; just needs explicit `http://` scheme.

1. `pc = Pinecone(api_key="pclocal", host=PINECONE_HOST)`; readiness retry loop: `pc.list_indexes()` up to 30×1 s (replaces the missing container healthcheck).
2. `if not pc.has_index(INDEX_NAME): pc.create_index(name, dimension=512, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"), deletion_protection="disabled")` — cloud/region required but ignored locally.
3. `host = pc.describe_index(INDEX_NAME).host` (e.g. `localhost:5081`); prefix `http://` if no scheme; `index = pc.Index(host=host)`. **Never hardcode 5081.** Same resolve logic lives in `clients/pinecone_client.py`, reused by `vector_tool` — one function, two callers.
4. Gzip-stream line-by-line; per record: drop junk metadata keys `{creator, producer, moddate, creationdate}`, `source` → basename, `company` = `COMPANY_BY_FILE[basename]` (`Alphabet_10K_FY2025.pdf→Google`, `Amazon→Amazon`, `Apple→Apple`, `Meta→Meta`; KeyError = fail loudly on unknown file). Upsert batches of 200, no namespace arg.
5. Assert `describe_index_stats().total_vector_count == 4072` (retry ≤15 s — stats can lag).

## E5. Backend module contracts

- **`main.py`**: `create_app(*, settings=None, graph=None, engine=None, agent_engine=None, pinecone_index=None)` — lifespan builds real clients only when args are None (tests inject stubs → offline, $0). CORS: `allow_origins=settings.cors_origins`, `allow_headers=["Authorization","Content-Type"]`, `expose_headers=["x-vercel-ai-ui-message-stream"]`. `GET /api/health` → `{postgres_rows, vector_count, ok}` where `ok = (192 and 4072)`.
- **`db.py`**: `build_engine(url)` (app role) vs `build_agent_ro_engine(url)` (separate engine, `connect_args={"options": "-c default_transaction_read_only=on"}`). The agent never receives the app engine.
- **`auth/service.py`** (pure, no FastAPI imports): `hash_password` (bcrypt, cost 12, raise ValueError >72 bytes) · `verify_password` · `create_access_token(*, sub, role, secret, expiry_min, now=None)` (PyJWT HS256, claims sub/role/iat/exp; `now` injectable for tests) · `decode_token(token, secret) -> TokenPayload{sub, role, exp}` (raises jwt errors; caller maps to 401).
- **`auth/models.py`**: `users(id UUID PK default uuid4, email unique, password_hash, display_name, role server_default 'user', created_at server_default now())`.
- **`auth/deps.py`**: `HTTPBearer(auto_error=False)`; `get_current_user` → 401 on missing/invalid/expired/unknown-sub; `require_role(role)` factory → 403.
- **`auth/router.py`**: `POST /api/auth/register` → 201 `{access_token, token_type:"bearer"}` | 409 dup email; `POST /api/auth/login` → 200 | 401 (identical message for unknown email vs bad password); `GET /api/auth/me`. `RegisterRequest{email: EmailStr, password: min 8, display_name: min 1}`.
- **`chat/schemas.py`**: must parse the AI SDK `DefaultChatTransport` POST body `{id, messages: UIMessage[], trigger, messageId}` — `UIMessageIn{id, role, parts:[{type, text?, extra allowed}]}` with `extra="ignore"` at top level; helpers `latest_user_text(req)` (422 if empty) and `to_history(req)`.
- **`chat/router.py`**: `POST /api/chat` (auth-guarded) → `StreamingResponse(stream_chat(...), media_type="text/event-stream", headers={"x-vercel-ai-ui-message-stream":"v1", "Cache-Control":"no-cache", "Connection":"keep-alive", "X-Accel-Buffering":"no"})`.
- **`agent/coverage.py`**: `CoverageMap{sql_years: dict[str, list[int]], vector_companies: frozenset}` built at startup; `ALIASES` (alphabet→Google, facebook→Meta, กูเกิล→Google, เฟซบุ๊ก/เฟสบุ๊ค→Meta, แอปเปิล→Apple, ไมโครซอฟท์→Microsoft, …) as **deterministic backstop** behind the LLM's world-knowledge normalization; `resolve_company(name, coverage) -> str | None`; `apply_gate(decision, coverage) -> GateResult{effective_route, companies, years, notes, clarification}` — order of checks: **intent=off_topic → refuse(scope template); intent=vague → clarify (question from route output)**; unknown company after alias+LLM normalization → refuse ("not in my data"); LLM flagged a mention unresolved/ambiguous → clarify with its candidates; all years out → refuse; some out → trim+note; vector/both company w/o 10-K → drop from vector half + note.
- **`agent/sql_tool.py`**: `validate_sql(sql, *, row_limit=100) -> str` — sqlglot(dialect="postgres"): exactly one statement, SELECT only, every `exp.Table` == `financial_data` (catches joins/subqueries/CTEs to users), no SELECT INTO, inject/cap LIMIT. `SqlTool(engine).run(sql) -> SqlResult{sql, rows, error}`.
- **`agent/vector_tool.py`**: `VectorTool(index, embed, *, top_k, score_floor).query(question, companies) -> VectorResult{chunks: [Chunk{id, company, source, page, text, score}], rejected: [{id, company, score}]}` — embed once, **one `index.query` per company** with `filter={"company": {"$eq": c}}, include_metadata=True`; `rejected` feeds debug.
- **`clients/`**: `build_llm` (temp 0, max_tokens ≈1200) · `build_embedder` · `resolve_index(settings)` (describe_index host resolution) · `vector_count(index)` for health.

## E6. LangGraph implementation

**Structured models** (strict-compatible: no dict fields, all required, Optional = null-union):

```python
class CompanyMention(BaseModel):
    mentioned: str                       # verbatim from the question ("เฟซบุ๊ก", "the iPhone company")
    canonical: Optional[str]             # LLM's world-knowledge normalization ("Meta", "Apple"); None = can't resolve
    confident: bool                      # False → gate sends to clarify with candidates

class RouteDecision(BaseModel):
    reasoning: str                       # FIRST — generation order matters on gpt-4o-mini-class
    intent: Literal["financial", "off_topic", "vague"]   # scope guardrail
    companies: list[CompanyMention]      # world-knowledge entity resolution happens HERE, in-call
    years: list[int]                     # lax coercion "2024"→2024 (mode="before"); range enforced by gate
    metrics: list[str]
    route: Literal["sql", "vector", "both", "refuse", "clarify"]
    clarification: Optional[str]         # the question to ask back; required (null) unless route/intent needs it
    language: str                        # field_validator: non-empty

class Citation(BaseModel):
    kind: Literal["sql", "chunk"]; source: Optional[str]; page: Optional[int]; quote: Optional[str]

class SynthesisEnvelope(BaseModel):
    reasoning: str        # FIRST (discarded from user-visible stream)
    answer: str           # SECOND — the streamed field (see E7)
    citations: list[Citation]  # LAST — arrives after answer closes; delivered via data parts
```

Route-node prompt contract for `companies`: *"Normalize every company mention to its canonical official name using your knowledge of real-world brands (e.g. Facebook/IG → Meta, the iPhone maker → Apple, Alphabet → Google's parent listed here as Google). If you cannot resolve a mention confidently, set canonical=null / confident=false — never guess silently."* The deterministic `ALIASES` map then re-checks each `canonical` (backstop for LLM drift; keeps tests deterministic). This is the production win: new phrasings/brands work without editing config, because resolution rides on model knowledge, while the small alias table pins the data-specific quirks (Google-not-Alphabet).

Both models bound `with_structured_output(Model, method="json_schema", strict=True, include_raw=True)`; in-node fail-closed re-ask (once, appending error text) → refuse.

**AgentState** (TypedDict, total=False): `question, history · route (RouteDecision dump), effective_route, companies (canonical), years, coverage_notes, refusal_reason, clarification · sql, sql_rows, computed (Python-calculated growth % — part of verify's allowed set), chunks, rejected_chunks · envelope, verify {ok, ungrounded, attempt}, verify_attempts, final_answer · debug`.

**Wiring**:

```python
g.set_entry_point("route")                                   # route node = LLM call + apply_gate together
g.add_conditional_edges("route", after_route,
    {"refuse": "refuse", "clarify": "clarify", "sql": "sql_retrieve", "vector": "vector_retrieve"})
    # "both" maps to "sql" (SQL half first); intent gate resolved inside apply_gate
g.add_conditional_edges("sql_retrieve", after_sql, {"vector": "vector_retrieve", "synthesize": "synthesize"})
g.add_edge("vector_retrieve", "synthesize")
g.add_edge("synthesize", "verify")
g.add_conditional_edges("verify", after_verify, {"ok": END, "retry": "synthesize", "fail": "refuse"})
    # retry iff not ok and verify_attempts == 1
g.add_edge("refuse", END)
g.add_edge("clarify", END)      # terminal: streams the clarifying question; user's reply is just the next turn
```

**Every node accepts and forwards `config: RunnableConfig`** (don't rely on contextvar propagation), adding its tag: route node LLM call tagged `"route"`, synthesize tagged `"synthesize"` — the SSE emitter filters `on_chat_model_stream` events on the `"synthesize"` tag. Call `graph.astream_events(state, version="v2")` **without** `include_tags` (emitter also needs `on_chain_end` node-lifecycle events; filter in-loop). `astream_events` auto-streams nested `ainvoke`'d chat models (verified LangChain behavior).

Event dispatch: `on_chat_model_stream` + tag synthesize → feed extractor → `text-delta` · `on_chain_end` name=route → `data-route` + `data-coverage` · name=sql_retrieve → `data-citations` (id `citations-1`) · name=vector_retrieve → update same id · name=verify → `text-end` + `data-verify` + reset extractor · name=refuse / name=clarify → template/question as a fresh text part.

**AnswerFieldExtractor** (the one non-obvious mechanism): synthesize streams strict JSON, so raw deltas are JSON fragments. A ~50-line pure state machine fed raw deltas: (1) scan for `"answer"` key then its opening quote (schema order guarantees `reasoning` streams first — discarded); (2) yield un-escaped chars (`\n`, `\"`, `\\`, `\uXXXX`) until the unescaped closing quote; (3) ignore `citations` (delivered via data parts). Fallback: extractor emitted nothing when node ends → emit `envelope.answer` as one `text-delta`. Pure → fully unit-testable offline.

## E7. SSE emitter wire format (AI SDK UI message stream v1, shapes verified against ai-sdk.dev)

Every line `data: <json>\n\n` with `json.dumps(..., ensure_ascii=False)` (Thai stays readable); terminator literal `data: [DONE]\n\n`. Happy path:

```
data: {"type":"start","messageId":"<uuid>"}
data: {"type":"start-step"}
data: {"type":"data-route","id":"route-1","data":{"route":"both","intent":"financial","companies":["Meta","Google"],"years":[2024,2025],"language":"th"}}
data: {"type":"data-coverage","id":"coverage-1","data":{"notes":["Microsoft has no 10-K …"]}}
data: {"type":"data-citations","id":"citations-1","data":{"sql_rows":[…],"chunks":[]}}
data: {"type":"data-citations","id":"citations-1","data":{…with chunks}}      ← same id → client reconciles
data: {"type":"text-start","id":"draft-1"}
data: {"type":"text-delta","id":"draft-1","delta":"Meta (Facebook) มี"}
data: {"type":"text-end","id":"draft-1"}
data: {"type":"data-verify","id":"verify-1","data":{"ok":true,"attempt":1,"ungrounded":[]}}
data: {"type":"finish-step"}
data: {"type":"finish","messageMetadata":{"route":"both","coverage_notes":[…],"citations":[…],"debug":{…}}}
data: [DONE]
```

**Clarify path**: `data-route` carries `route:"clarify"` (RouteBadge renders a "needs info" state); the clarifying question streams as an ordinary text part; no citations, no verify part; `finish` as usual. The user's answer is simply the next `sendMessage` — full history rides along, so the route node sees the clarified context.

**Veto path**: server guarantees *correct final text is always the LAST text part*; frontend renders only the last text part. On `data-verify {ok:false, attempt:1, ungrounded:["112,010"]}` the graph loops verify→synthesize; regenerated tokens stream as **new part `draft-2`**; second verify reconciles same id `verify-1`. Second failure → refusal template streams as `draft-3` → finish. Frontend rules: (a) render last text part only; (b) provisional styling until `data-verify ok:true` or stream end; (c) final `ok:false` → "showing grounded portion only" badge. `debug` = `{route_decision (incl. intent + mention resolutions), gate_result, sql, per_company_scores incl. rejected, verify}` duplicated in finish metadata. Mid-stream exception → `data: {"type":"error","errorText":"…"}` then `[DONE]`.

## E8. Frontend contracts

```tsx
const { messages, sendMessage, status } = useChat({
  transport: new DefaultChatTransport({
    api: `${process.env.NEXT_PUBLIC_API_URL}/api/chat`,
    headers: () => ({ Authorization: `Bearer ${getToken() ?? ''}` }),  // function form → fresh token per request
  }),
});
// data parts read from message.parts (no transient parts used → onData unnecessary):
// find('data-route') → RouteBadge (SQL / 10-K / Hybrid / Refused / Needs-info) · find('data-coverage') → notes banner
// find('data-citations') → CitationList · find('data-verify') → verify badge
// filter('text').at(-1) → rendered text (veto rule) · provisional = status === 'streaming' && !verify
```

Pages: `/` redirect → `/chat`; `/login` + `/register` share `<AuthForm mode>`; `/chat` is `'use client'` with `useEffect` token guard (client guard sufficient — token in localStorage, invisible to middleware; trade-off in README). `lib/auth.ts` get/set/clearToken (key `smc_token`); `lib/api.ts` `apiFetch` injects bearer, 401 → clear + redirect. shadcn: `init -d` then `add button input card badge collapsible table alert skeleton scroll-area label`.

## E9. Init commands & Makefile

```bash
uv init backend --python 3.12
cd backend && uv add fastapi "uvicorn[standard]" sqlalchemy "psycopg[binary]" pyjwt bcrypt \
  pinecone "langchain-openai>=0.3" langgraph sqlglot pydantic-settings "pydantic[email]" httpx
uv add --dev pytest pytest-asyncio

npx create-next-app@latest frontend --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*" --use-npm
cd frontend && npx shadcn@latest init -d && npx shadcn@latest add button input card badge collapsible table alert skeleton scroll-area label
npm i ai @ai-sdk/react
```

Makefile: `up` = `docker compose up -d` · `seed` = `uv run --project backend python scripts/load_pinecone.py` · `api` = `uvicorn app.main:app --reload --port 8000` · `web` = `npm --prefix frontend run dev` · `test` = pytest · `eval` = eval_baseline · `down` = `docker compose down -v` (**-v is load-bearing**: forces initdb re-run). README lists raw commands beside each target (Windows without make). Scripts run under the backend uv project — one environment.

## E10. Test matrix (all offline via stub injection)

| File | Key cases |
|---|---|
| `test_auth.py` | hash/verify roundtrip; wrong pw False; >72-byte raises; token roundtrip; expired (injected `now`); tampered sig; register 201/dup 409; login wrong pw 401 (same msg as unknown email); /me ± token; /api/chat no token → 401 |
| `test_coverage.py` | parametrized over `fixtures/routing_cases.json`; `build_coverage` on seeded test engine (BlackRock=[2022,2023], Shopify=[2024,2025]) |
| `test_sql_validator.py` | accepts plain/aggregate/WHERE SELECTs, injects LIMIT, caps >100; rejects UPDATE/DELETE/DROP, multi-statement, `FROM users`, join/subquery/CTE to users, SELECT INTO |
| `test_router.py` | valid decision → correct edge; `parsed=None` once → re-ask succeeds (stub called 2×); twice → refuse + reason; `parsing_error` identical; **intent=off_topic → refuse edge; intent=vague → clarify edge + clarification text surfaced; unconfident mention → clarify with candidates** |
| `test_verify.py` | grounded variants pass (`99,803`/`99803`/`$99,803`; `22.2%` vs computed `0.2224`; one-decimal rounding); fabricated → ok:false + listed; second failure → "fail" edge |
| `test_schemas.py` | `model_json_schema()` strict-compat: no bare dicts, all required, `additionalProperties:false` (incl. nested `CompanyMention`); validators (empty language rejected, `"2024"` coerced) |
| `test_sse.py` | AnswerFieldExtractor: plain/escaped/`\uXXXX` Thai/split-across-chunks/empty; emitter over scripted fake event stream asserts exact part ordering + veto sequence (draft-2, reconciled verify-1) + **clarify sequence (data-route clarify → text part → finish, no verify part)** |

`routing_cases.json` case = `{id, question, route_decision (stubbed LLM output — gate input), expected: {effective_route, companies, years, coverage_notes_contains, clarification_expected}}`. Required cases: Facebook→Meta (EN+TH), Alphabet→Google, กูเกิล→Google, แอปเปิล→Apple, BlackRock-2025 trim/refuse, Shopify-2022 ditto, Siemens refuse, Microsoft-both vector-half-drop + note, year-2021 trimmed + note, **off-topic ("write me a poem" / "แนะนำร้านอาหาร") → refuse(scope), vague ("how's the company doing?") → clarify, fuzzy-resolved ("the iPhone company" → Apple, confident) → proceeds, ambiguous (confident=false) → clarify with candidates**.

## E11. Commit plan (conventional commits on `dev`)

- **Day 1**: `chore: add docker compose data stack` → `feat(db): agent_ro read-only role` → `feat(scripts): pinecone loader (host resolution, metadata normalization)` → `feat(api): app factory, settings, health` → `chore: makefile + .env.example`
- **Day 2**: `feat(auth): model, bcrypt+jwt service, register/login/me` → `test(auth)` → `feat(web): next.js + shadcn scaffold` → `feat(web): auth pages, token storage, guarded chat shell` → `feat(api): stub SSE endpoint (UI message stream v1)` → `feat(web): useChat via DefaultChatTransport`
- **Day 3**: `feat(agent): coverage map + aliases` → `test(agent): routing fixtures` → `feat(agent): route node (intent gate, world-knowledge naming, strict structured output, fail-closed)` → `feat(agent): clarify node` → `feat(agent): sql tool + validator` → `test(agent): sql rejection matrix` → `feat(agent): vector tool (per-company queries)` → `feat(agent): graph wiring + answer-field SSE streaming` → `feat(agent): synthesis envelope`
- **Day 4**: `feat(agent): verify node + stream-veto` → `test(agent): verify variants` → `feat(agent): hybrid path + refusal/scope templates` → `feat(web): RouteBadge, CitationList, verify states` → `feat(api): debug in finish metadata` → `fix(agent): tune score floor`
- **Day 5**: `feat(scripts): eval_baseline over SSE` → `docs: README` → `chore: fresh-clone dry-run fixes`

## Verified sources (execution layer)

UI message stream v1 shapes + data-part reconciliation + DefaultChatTransport: [stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol), [streaming data](https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data), [chatbot docs](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot) · Pinecone SDK `has_index`/`create_index`/host resolution + pinecone-local constraints: [Python SDK](https://docs.pinecone.io/reference/python-sdk), [local development](https://docs.pinecone.io/guides/operations/local-development) · `astream_events` v2 tag filtering + nested auto-streaming: [LangGraph streaming](https://docs.langchain.com/oss/python/langgraph/streaming), [astream_events reference](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/astream_events)
