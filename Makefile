.PHONY: help up down restart logs ps build shell test test-unit test-integration coverage coverage-html load db-shell format clean ingest health

help:
	@echo "Available commands:"
	@echo "  make up               - Build and start all services"
	@echo "  make down             - Stop all services"
	@echo "  make restart          - Restart all services"
	@echo "  make logs             - Follow service logs"
	@echo "  make ps               - Show running services"
	@echo "  make build            - Build Docker images"
	@echo "  make shell            - Open shell in api container"
	@echo "  make db-shell         - Open psql shell in postgres container"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests"
	@echo "  make test-integration - Run integration tests"
	@echo "  make coverage         - Run tests with XML coverage report"
	@echo "  make coverage-html    - Run tests with HTML coverage report"
	@echo "  make load             - Run load tests with Locust"
	@echo "  make ingest           - Run sample ingestion script"
	@echo "  make health           - Check local health endpoint"
	@echo "  make clean            - Stop services and remove volumes"

up:
	docker compose up --build -d

down:
	docker compose down

restart:
	docker compose down
	docker compose up --build -d

logs:
	docker compose logs -f

ps:
	docker compose ps

build:
	docker compose build

shell:
	docker compose exec api bash

db-shell:
	docker compose exec postgres psql -U $$POSTGRES_USER -d $$POSTGRES_DB

test:
	docker compose exec api pytest -v

test-unit:
	docker compose exec api pytest tests/unit -v

test-integration:
	docker compose exec api pytest tests/integration -v

coverage:
	docker compose exec api sh -c "mkdir -p reports && pytest -v --cov=src --cov-report=term-missing --cov-report=xml:reports/coverage.xml --cov-fail-under=60"

coverage-html:
	docker compose exec api sh -c "mkdir -p reports && pytest -v --cov=src --cov-report=term-missing --cov-report=html:reports/htmlcov --cov-report=xml:reports/coverage.xml --cov-fail-under=60"

load:
	docker compose exec api locust -f tests/load/locustfile.py

ingest:
	docker compose exec api python scripts/ingest_sample_data.py

health:
	curl -f http://localhost:8000/api/v1/health

clean:
	docker compose down -v