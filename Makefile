.PHONY: up seed api web test eval down

up:
	docker compose up -d

seed:
	uv run --project backend python scripts/load_pinecone.py

api:
	cd backend && uv run uvicorn app.main:app --reload --port 8000

web:
	npm --prefix frontend run dev

test:
	cd backend && uv run pytest

eval:
	uv run --project backend python scripts/eval_baseline.py

down:
	docker compose down -v
