.PHONY: up seed api web test eval eval-ragas down

up:
	docker compose up -d

seed:
	uv run --project backend python scripts/load_pinecone.py
	uv run --project backend python scripts/load_chunk_text.py

api:
	cd backend && uv run uvicorn app.main:app --reload --port 8000

web:
	npm --prefix frontend run dev

test:
	cd backend && uv run pytest

eval:
	uv run --project backend python scripts/eval_baseline.py

eval-ragas:
	uv run --project backend --extra eval python scripts/eval_ragas.py

down:
	docker compose down -v
