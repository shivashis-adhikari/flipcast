PY      ?= .venv/bin/python
DSN     ?= postgresql://postgres@localhost:5433/postgres
PG_DUMP ?= docker compose exec -T postgres pg_dump

.PHONY: dev lint test itest image up down demo clean

dev:
	python3 -m venv .venv
	$(PY) -m pip install -e '.[dev]'

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
	$(PY) -m mypy

test:
	$(PY) -m pytest -q tests/unit

# needs `make up` (or any PG 18 with flipcast_sizes) first
itest:
	FLIPCAST_TEST_DSN=$(DSN) FLIPCAST_TEST_PG_DUMP="$(PG_DUMP)" $(PY) -m pytest -q tests/integration

image:
	docker build -t flipcast-postgres:18 -f docker/Dockerfile .

up:
	docker compose up -d --wait

down:
	docker compose down

demo: up
	$(PY) -m flipcast demo --dsn $(DSN) --dir flipcast-demo --pg-dump "$(PG_DUMP)"
	$(PY) -m flipcast -c flipcast-demo/flipcast.toml backtest
	$(PY) -m flipcast -c flipcast-demo/flipcast.toml forecast

clean:
	rm -rf flipcast-demo build dist src/*.egg-info .pytest_cache .mypy_cache .ruff_cache
