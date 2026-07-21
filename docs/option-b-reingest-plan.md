# Option B (custom re-ingestion) — an evidence-driven plan

*(Not yet built. Future-roadmap design doc, cross-linked from
`docs/implementation-plan.md` §Future improvements #2 and README's Future
roadmap. Every number below was measured directly against
`data/pinecone_vectors.jsonl.gz` and the four PDFs in `10k_filings/` — see
"What the data actually looks like". Re-read the current state of
`backend/app/agent/` and `scripts/` before implementing; this repo is
edited by multiple sessions in parallel.)*

## Where this comes from

`Take-Home Task.pdf` §6 offers two ways to populate the vector store:

> 1. **Option A, use the fixture.** Upsert `data/pinecone_vectors.jsonl.gz`
>    directly. Text extracted per page (PyPDF), split with a recursive
>    character splitter (`chunk_size=1000, overlap=200`), embedded with
>    `text-embedding-3-small` at `dimensions=512`.
> 2. **Option B, re-ingest.** If you want to control chunk size or embedding
>    dimension, ignore the fixture and build the index from the raw PDFs.

The app ships **Option A** as the only path today, and that stays the
default (per `CLAUDE.md`: provided data is loaded as-is, never regenerated).
The Day-3 live smoke test (`agent-output/day3-smoke-test-findings.md`) found
a concrete gap, though: for qualitative "why did revenue grow / compare
strategy" questions, top-ranked chunks were often risk-factor or
financial-statement text rather than the MD&A / business narrative that
actually answers the question. Everything shipped since (hybrid dense +
lexical fusion, cross-encoder reranking, `is_boilerplate` filtering) improves
the *ranking* of what's indexed. This plan attacks the *ingestion* layer.

## What the data actually looks like (measured, not assumed)

Cracking open the fixture and the PDFs first changed the plan materially.
Findings:

| Fact | Value | Why it matters |
|---|---|---|
| Vectors in fixture | **4,072**, all `512`-dim, single `__default__` namespace | Matches `EXPECTED_COUNT` in both loaders. |
| Metadata keys present | `page, page_label, source, title, total_pages` (+ PDF junk `creator/producer/moddate/creationdate`, already dropped by `load_pinecone.py`'s `DROP_KEYS`) | **No `item_section`, no `company`** — `company` is derived from the filename by the loaders (`Alphabet_10K_FY2025.pdf → "Google"`). Section is simply absent. |
| Chunk text length | min 152 / median 925 / max 999 chars | Confirms `chunk_size≈1000`. |
| Chunks per (company, page) | min 2, max 18, **avg 7.2** | **Chunking is within-page — a chunk never spans two pages.** This is the key enabler (see Tier 1). |
| Chunks starting mid-sentence (lowercase first char) | **50%** (2,020/4,072) | Fixed-character splitting cuts through sentences. |
| Chunks ending with no terminal punctuation | **81%** (3,304/4,072) | Same artifact at the tail. Weak citation/embedding units. |
| Most-repeated line | `'4/20/26, 12:05 PM meta-20251231'` ×458; `'Table of Contents'` ×424 | Print-to-PDF running headers are **baked into the embeddings**, not just present as separate rows. |

The `source` path in the raw fixture is the original author's machine
(`/Users/sainytk/.../Meta_10K_FY2025.pdf`) — the loaders already normalize
it to the bare filename, so nothing here depends on that path.

### Section structure is recoverable — but not the way the naive plan assumed

The naive assumption ("scan each chunk for an `Item N.` heading and tag it")
is **wrong**, and the data proves it. The token `Item 7.` appears in only
~40–48 chunks per filing, and each Item label appears a suspiciously uniform
16–26 times — because those are almost all **table-of-contents rows and
Part-divider cross-references**, not body headings. The *body* of MD&A does
not repeat the string "Item 7" on every chunk. So a chunk cannot be tagged
by looking inside it.

What *does* work: reconstruct **page ranges**. Body headings sit cleanly at
line starts (measured: **422 line-start occurrences vs. 2 mid-line**), on
monotonically increasing pages. Example (Apple, PDF page index):

```
Item 1 @p3  1A@p8  1B/1C/2@p23  3/4@p24  5@p26  6@p27  7(MD&A)@p28
7A@p37  8@p38  9/9A@p67  10–14@p68  15@p69  16@p74
```

The false positives (cross-references like Alphabet's "Item 1 …" at p64,
*after* Item 7 at p49; Meta's "Item 8 …" at p117 before the real Item 8 at
p135) all **break the canonical Item ordering**, so a greedy monotonic scan
(accept a heading only if its page is non-decreasing *and* its Item rank
strictly advances) discards them. A ToC cross-check is free if wanted: the
printed `page_label` is a constant offset from the PDF `page` (measured:
`label = page + 1` for all 604 Apple chunks), so the ToC's "Item 7 … 28"
independently confirms the body scan.

**Result of running that scan + assigning every chunk to a section by its
page (proven end-to-end against the fixture):**

| Company | ToC pg | chunks tagged | Business (1) | Risk (1A) | MD&A (7) | Fin. stmts (8) |
|---|---|---|---|---|---|---|
| Apple | 2 | 586 / 604 | 50 | 182 | 54 | 186 |
| Amazon | 2 | 832 / 848 | 40 | 176 | 136 | 370 |
| Google | 3 | 1028 / 1052 | 82 | 234 | 154 | 398 |
| Meta | 2 | 1512 / 1568 | 76 | **570** | 184 | 444 |
| **All** | | **3,958 / 4,072 (97%)** | | | | |

The untagged 3% is exactly the cover + ToC pages *before* Item 1 — correctly
left `None`. **This tagging required zero re-embedding.**

### The base-rate problem, quantified

This table *is* the smoke-test bug's mechanism. Meta has **570 Risk-Factor
chunks vs. 184 MD&A chunks** — a 3:1 ratio. Financial-statement tables
(Item 8) are the single largest section for every filing. A section-blind
retriever surfaces risk/legal/table text for a "why did revenue grow"
question largely because there is simply *more of it* to match. Section
awareness corrects the base rate directly.

## The reframe: two tiers, not one

The original one-paragraph roadmap bullet lumped everything into "re-ingest
from PDFs into a new index." The measurements split it into two levers with
very different cost/risk:

- **Tier 1 — section-metadata backfill. No re-embedding.** Because chunking
  is within-page, every existing fixture chunk can be tagged with its Item
  section from its `page` alone. This is the big retrieval win, it keeps the
  provided embeddings byte-for-byte (honoring "load as-is"), costs **$0**,
  and is fully reversible. **This is where the value is.** It is arguably not
  even "Option B" — it's an *enrichment* of the Option-A fixture.
- **Tier 2 — true re-ingestion from PDFs.** Only needed for the levers Tier 1
  can't touch: fixing the 50%-mid-sentence chunk boundaries, stripping the
  baked-in running headers *before* embedding, and changing `chunk_size` /
  embedding `dimension` (the literal §6 Option-B ask). This does require
  re-extraction + re-embedding + a separate index.

Do Tier 1 first; measure; only spend Tier 2 effort if the eval still shows a
gap Tier 1 didn't close.

---

## Tier 1 — section-metadata backfill (no re-embed)

### Where the code goes

The `page → item_section` map is a small deterministic derivation. It slots
into the **two existing loaders**, right next to the `company` derivation
they already do — not a new pipeline:

1. **`scripts/section_map.py`** (new, ~60 lines): given the fixture, build
   `{company: {page: item_section}}` using the monotonic-scan algorithm
   above. Pure function, unit-testable against the four known filings (the
   page-range table above is the golden fixture for its tests).
2. **`scripts/load_pinecone.py`**: in `normalize_record`, after setting
   `metadata["company"]`, set
   `metadata["item_section"] = SECTION_MAP[company].get(page)` (and
   optionally `item_title`). Metadata-only change; `values` untouched, so
   Pinecone `upsert` overwrites just the metadata of the same ids — the
   embeddings never move. `EXPECTED_COUNT` and dimension are unchanged.
3. **`scripts/load_chunk_text.py` + `scripts/initdb/02_chunk_text.sql`**: add
   a nullable `item_section TEXT` column (additive migration; existing rows
   stay valid as `NULL`), and set it in the insert alongside `company`.

Both loaders already share `COMPANY_BY_FILE` and read the same
`data/pinecone_vectors.jsonl.gz`; the section map keys off the same
`(company, page)` they already compute, so the two sinks stay consistent by
construction and `hybrid_tool.py`'s by-id fusion needs no change.

### Retrieval-side payoff

Once `item_section` exists on both sinks, add an optional
`item_sections: list[str] | None` param to `vector_tool.py` /
`text_search_tool.py`:

- **Soft boost (preferred default):** up-weight matching-section chunks
  during RRF fusion rather than hard-excluding others — preserves recall when
  the section map is `None` or a relevant chunk sits just outside the
  boosted set. Hard metadata filter (`{"item_section": {"$in": [...]}}` on
  Pinecone, `WHERE item_section = ANY(...)` on `chunk_text`) is available for
  the cases where precision clearly matters more.
- **Where the hint comes from:** the router already extracts intent/metrics
  into `RouteDecision` (`agent/schemas.py`). Derive the section hint from a
  lightweight keyword map rather than a new LLM call — e.g. "why / strategy /
  grew / drivers" → `[Item 1, Item 7]`; "risk / lawsuit / litigation" →
  `[Item 1A, Item 3]`. Falls through to no hint (today's behavior) when
  nothing matches.

### Cost / risk

Effectively free (metadata rewrite of 4,072 records; no OpenAI calls). Risk
is limited to section-detection edge cases, all of which fail safe to
`item_section=None` (chunk stays fully retrievable as it is today). Known
edge cases already seen in the data: Item 6 is `[Reserved]` (empty, fine);
Meta's Item 4 lands unusually late (p100) but still monotonic; the ToC page
is auto-detected as the page hosting the most distinct Item labels and
excluded. Because it's metadata-only, rollback is a re-run of the current
loaders.

---

## Tier 2 — true re-ingestion (only if Tier 1's eval leaves a gap)

This is the literal §6 Option B, and the only path that can fix what's baked
into the fixture's vectors.

### What it buys that Tier 1 can't

- **Sentence-aware boundaries** — 50% of current chunks start mid-sentence,
  81% end mid-sentence. Re-splitting with a recursive splitter that respects
  paragraph/sentence separators (and **never crosses a detected section
  boundary**) produces cleaner embedding + citation units.
- **Header stripping before embedding** — remove the `'4/20/26, 12:05 PM
  meta-20251231'` / `'Table of Contents'` running headers (458× / 424×) at
  extraction time, so the noise is never embedded, not just filtered
  downstream by `is_boilerplate` (which can only drop whole contaminated
  chunks, never clean the vector).
- **Tunable `chunk_size` / `dimension`** — the explicit §6 ask.

### Sketch: `scripts/reingest_10k.py`

Mirrors the existing loaders' shape (idempotent, stable ids, assert-on-count):

1. **Extract** per page from `10k_filings/*.pdf`. Start with PyPDF (it
   produced the fixture, so results are comparable) + a regex header-strip
   pass reusing `agent/vector_tool.py`'s `_HEADER_NOISE`. A half-hour spike
   comparing PyPDF vs. `pymupdf`/`pdfplumber` layout-aware extraction on
   Meta's filing (the largest, 215 pages) decides whether layout info is
   worth the dependency — leave it as a spike, not a guess.
2. **Detect sections** with the *same* monotonic page-range scan from Tier 1
   (reuse `scripts/section_map.py`), fail-open to `None`.
3. **Chunk** with configurable `chunk_size`/`overlap`, never crossing a
   section boundary.
4. **Embed** via the existing `openai_embed_model` / `embed_dimensions`
   config knobs.
5. **Load** to a **separate** Pinecone index + `chunk_text`-shaped table,
   with ids namespaced so they can't collide with the fixture
   (`reingest:{company}:{item}:{n}`).

### Toggle — keep Option A the untouched default

Add to `config.py` (both default off): `pinecone_index_reingested: str` and
`use_reingested_index: bool = False`, flipping which index/table
`main.py` wires up. Different embedding dimension ⇒ **must** be a separate
index (Pinecone forbids mixed dimensions in one index). Default false ⇒ app
behavior byte-for-byte unchanged unless opted in.

### Cost / risk

Re-embedding ~4k+ chunks at `text-embedding-3-small` is a few cents (well
under the $10 cap) but non-zero, unlike Option A. The real risk is
extraction/section-detection accuracy on inconsistent real-world 10-K
formatting — ship with the fail-open `None` fallback, never block on perfect
detection.

## Evaluation — prove it before shipping either tier

Run the existing tooling twice (baseline vs. enriched) via config swap:

- **`scripts/eval_ragas.py`** — faithfulness / context-precision /
  context-recall, diffed.
- **`scripts/eval_advanced.py`** — adversarial + naturalness suite.
- **The concrete acceptance test:** re-run the smoke-test's Q2 (compare
  Google/Meta revenue structure & strategy) and Q3 (why did revenue grow)
  and manually diff retrieved chunks — did MD&A/Business prose (Item 1/7)
  displace the risk-factor/table chunks (Item 1A/8) that Tier 1's base-rate
  table predicts were crowding them out? That is the direct, falsifiable bar.

## Suggested sequencing

Tier 1 is cheap, safe, and high-leverage — it slots in right after the
faithfulness gate, ahead of the semantic-layer / ops items, at roughly a day
(section-map module + tests + two loader diffs + retrieval boost + eval).
Tier 2 is a "few days" item and **conditional on Tier 1's eval still showing
a gap**. Neither is required for the baseline acceptance questions — those
already pass on Option A + hybrid + rerank; this raises the quality ceiling
for open-ended qualitative questions.
