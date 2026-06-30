.PHONY: all proto build test run clean docker-build docker-up docker-down migrate help

# Default target
all: proto build

# Generate protobuf code
proto:
	@echo "Generating protobuf code..."
	@mkdir -p gen/go
	protoc --proto_path=proto \
		--go_out=gen/go --go_opt=paths=source_relative \
		--go-grpc_out=gen/go --go-grpc_opt=paths=source_relative \
		proto/common/common.proto \
		proto/user/user.proto \
		proto/auth/auth.proto \
		proto/billing/billing.proto \
		proto/notification/notification.proto

# Build all services
build: build-user build-auth build-billing

build-user:
	@echo "Building user-service..."
	cd services/user-service && go build -o ../../bin/user-service ./cmd/main.go

build-auth:
	@echo "Building auth-service..."
	cd services/auth-service && go build -o ../../bin/auth-service ./cmd/main.go

build-billing:
	@echo "Building billing-service..."
	cd services/billing-service && go build -o ../../bin/billing-service ./cmd/main.go

build-notification:
	@echo "Building notification-service..."
	cd services/notification-service && pip install -r requirements.txt

# Run tests
test:
	@echo "Running tests..."
	cd services/user-service && go test -v ./...
	cd services/auth-service && go test -v ./...
	cd services/billing-service && go test -v ./...

# Start infrastructure (databases, redis, etc.)
infra-up:
	docker-compose up -d postgres-user postgres-auth postgres-billing redis nats jaeger prometheus grafana

# Stop infrastructure
infra-down:
	docker-compose down

# Start Kong API Gateway
kong-up:
	docker-compose up -d kong-database
	@echo "Waiting for Kong database..."
	@sleep 5
	docker-compose up -d kong-migrations
	@echo "Waiting for Kong migrations..."
	@sleep 10
	docker-compose up -d kong
	@echo "Kong is starting..."
	@sleep 5
	@echo "Kong Admin GUI: http://localhost:8002"

# Stop Kong
kong-down:
	docker-compose stop kong kong-migrations kong-database

# Setup Kong routes and plugins
kong-setup:
	@echo "Setting up Kong routes and plugins..."
	./scripts/setup-kong.sh

# View Kong configuration
kong-status:
	@echo "Services:"
	@curl -s http://localhost:8001/services | jq '.data[] | {name, host, port}'
	@echo "\nRoutes:"
	@curl -s http://localhost:8001/routes | jq '.data[] | {name, paths}'
	@echo "\nPlugins:"
	@curl -s http://localhost:8001/plugins | jq '.data[] | {name, enabled}'

# Add JWT credential to Kong
kong-jwt-setup:
	@echo "Adding JWT credential to Kong..."
	@curl -s -X POST http://localhost:8001/consumers/microservices-platform/jwt \
		-F "key=microservices-platform" \
		-F "algorithm=RS256" \
		-F "rsa_public_key=@services/auth-service/keys/public.pem"
	@echo "\nJWT credential added"

# Full Kong setup (start + configure)
kong-full: kong-up
	@echo "Waiting for Kong to be ready..."
	@sleep 10
	$(MAKE) kong-setup

# Start all services
docker-up:
	docker-compose --profile services up -d

# Stop all services
docker-down:
	docker-compose --profile services down

# Build Docker images
docker-build:
	docker-compose build

# Run database migrations
migrate-user:
	@echo "Running user-service migrations..."
	@docker-compose exec postgres-user psql -U userservice -d users -f /migrations/001_initial_schema.up.sql

migrate-auth:
	@echo "Running auth-service migrations..."
	@docker-compose exec postgres-auth psql -U authservice -d auth -f /migrations/001_initial_schema.up.sql

migrate-billing:
	@echo "Running billing-service migrations..."
	@docker-compose exec postgres-billing psql -U billingservice -d billing -f /migrations/001_initial_schema.up.sql

migrate-all: migrate-user migrate-auth migrate-billing

# Generate JWT keys for development
generate-keys:
	@echo "Generating JWT keys..."
	@mkdir -p services/auth-service/keys
	openssl genrsa -out services/auth-service/keys/private.pem 2048
	openssl rsa -in services/auth-service/keys/private.pem -pubout -out services/auth-service/keys/public.pem
	@echo "Keys generated in services/auth-service/keys/"

# Clean build artifacts
clean:
	rm -rf bin/
	rm -rf gen/

# Install dependencies
deps:
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	cd services/user-service && go mod download
	cd services/auth-service && go mod download
	cd services/billing-service && go mod download
	cd services/notification-service && pip install -r requirements.txt

# Format code
fmt:
	cd services/user-service && go fmt ./...
	cd services/auth-service && go fmt ./...
	cd services/billing-service && go fmt ./...

# Lint code
lint:
	cd services/user-service && go vet ./...
	cd services/auth-service && go vet ./...
	cd services/billing-service && go vet ./...

# View logs
logs:
	docker-compose logs -f

logs-user:
	docker-compose logs -f user-service

logs-auth:
	docker-compose logs -f auth-service

logs-billing:
	docker-compose logs -f billing-service

logs-notification:
	docker-compose logs -f notification-service

# View Kong logs
logs-kong:
	docker-compose logs -f kong

# Help
help:
	@echo "Available targets:"
	@echo "  all            - Generate protos and build services"
	@echo "  proto          - Generate protobuf code"
	@echo "  build          - Build all services"
	@echo "  test           - Run tests"
	@echo "  infra-up       - Start infrastructure (databases, redis, etc.)"
	@echo "  infra-down     - Stop infrastructure"
	@echo "  docker-up      - Start all services with Docker"
	@echo "  docker-down    - Stop all services"
	@echo "  docker-build   - Build Docker images"
	@echo "  generate-keys  - Generate JWT keys for development"
	@echo "  deps           - Install dependencies"
	@echo "  clean          - Clean build artifacts"
	@echo "  fmt            - Format code"
	@echo "  lint           - Lint code"
	@echo "  logs           - View all logs"
	@echo ""
	@echo "Kong API Gateway:"
	@echo "  kong-up        - Start Kong API Gateway"
	@echo "  kong-down      - Stop Kong"
	@echo "  kong-setup     - Configure Kong routes and plugins"
	@echo "  kong-full      - Start Kong and configure it"
	@echo "  kong-status    - View Kong configuration"
	@echo "  kong-jwt-setup - Add JWT credential to Kong"
	@echo "  logs-kong      - View Kong logs"
	@echo ""
	@echo "  help           - Show this help"
