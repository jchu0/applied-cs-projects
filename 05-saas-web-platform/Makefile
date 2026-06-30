.PHONY: help install dev build test lint clean docker-up docker-down migrate

help:
	@echo "SaaS Platform - Development Commands"
	@echo ""
	@echo "Usage: make [command]"
	@echo ""
	@echo "Commands:"
	@echo "  install       Install all dependencies"
	@echo "  dev           Start development servers"
	@echo "  build         Build for production"
	@echo "  test          Run all tests"
	@echo "  lint          Run linters"
	@echo "  clean         Clean build artifacts"
	@echo "  docker-up     Start Docker containers"
	@echo "  docker-down   Stop Docker containers"
	@echo "  migrate       Run database migrations"

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

dev:
	docker-compose up -d db redis
	@echo "Starting backend..."
	cd backend && python manage.py runserver &
	@echo "Starting frontend..."
	cd frontend && npm run dev

build:
	cd frontend && npm run build
	cd backend && python manage.py collectstatic --noinput

test:
	cd backend && pytest
	cd frontend && npm test

lint:
	cd backend && ruff check . && black --check .
	cd frontend && npm run lint

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name node_modules -exec rm -rf {} +
	find . -type d -name .next -exec rm -rf {} +
	rm -rf backend/staticfiles

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f

migrate:
	cd backend && python manage.py migrate

makemigrations:
	cd backend && python manage.py makemigrations

createsuperuser:
	cd backend && python manage.py createsuperuser

shell:
	cd backend && python manage.py shell
