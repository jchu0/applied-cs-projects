.PHONY: all build clean proto test lint run-% docker-build docker-up docker-down migrate-% help

# Variables
GOPATH := $(shell go env GOPATH)
GOBIN := $(GOPATH)/bin
PROTOC := protoc
PROTO_DIR := proto
GEN_DIR := pkg/pb

# Services
SERVICES := auth-service user-service billing-service notification-service gateway

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

help:
	@echo "$(GREEN)Microservice Platform - Build Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Build Commands:$(NC)"
	@echo "  make build           - Build all services"
	@echo "  make build-auth      - Build auth service"
	@echo "  make build-user      - Build user service"
	@echo "  make build-billing   - Build billing service"
	@echo "  make build-notify    - Build notification service"
	@echo "  make build-gateway   - Build API gateway"
	@echo ""
	@echo "$(YELLOW)Proto Commands:$(NC)"
	@echo "  make proto           - Generate protobuf code"
	@echo "  make proto-deps      - Install protoc dependencies"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make run-auth        - Run auth service"
	@echo "  make run-user        - Run user service"
	@echo "  make run-billing     - Run billing service"
	@echo "  make run-notify      - Run notification service"
	@echo "  make run-gateway     - Run API gateway"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make docker-build    - Build Docker images"
	@echo "  make docker-up       - Start all services with docker-compose"
	@echo "  make docker-down     - Stop all services"
	@echo ""
	@echo "$(YELLOW)Database:$(NC)"
	@echo "  make migrate-up      - Run all migrations"
	@echo "  make migrate-down    - Rollback all migrations"
	@echo "  make migrate-auth    - Run auth service migrations"
	@echo "  make migrate-user    - Run user service migrations"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test            - Run all tests"
	@echo "  make test-coverage   - Run tests with coverage"
	@echo "  make lint            - Run linter"

# Install protoc dependencies
proto-deps:
	@echo "$(GREEN)Installing protoc dependencies...$(NC)"
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	go install github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-grpc-gateway@latest
	go install github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-openapiv2@latest

# Generate protobuf files
proto:
	@echo "$(GREEN)Generating protobuf files...$(NC)"
	@mkdir -p $(GEN_DIR)
	$(PROTOC) -I $(PROTO_DIR) \
		-I $(GOPATH)/pkg/mod/github.com/grpc-ecosystem/grpc-gateway@v1.16.0/third_party/googleapis \
		--go_out=$(GEN_DIR) --go_opt=paths=source_relative \
		--go-grpc_out=$(GEN_DIR) --go-grpc_opt=paths=source_relative \
		--grpc-gateway_out=$(GEN_DIR) --grpc-gateway_opt=paths=source_relative \
		$(PROTO_DIR)/common/v1/*.proto \
		$(PROTO_DIR)/auth/v1/*.proto \
		$(PROTO_DIR)/user/v1/*.proto \
		$(PROTO_DIR)/billing/v1/*.proto \
		$(PROTO_DIR)/notification/v1/*.proto

# Build all services
build: $(addprefix build-,auth user billing notify gateway)
	@echo "$(GREEN)All services built successfully!$(NC)"

build-auth:
	@echo "$(GREEN)Building auth-service...$(NC)"
	go build -o bin/auth-service ./cmd/auth-service

build-user:
	@echo "$(GREEN)Building user-service...$(NC)"
	go build -o bin/user-service ./cmd/user-service

build-billing:
	@echo "$(GREEN)Building billing-service...$(NC)"
	go build -o bin/billing-service ./cmd/billing-service

build-notify:
	@echo "$(GREEN)Building notification-service...$(NC)"
	go build -o bin/notification-service ./cmd/notification-service

build-gateway:
	@echo "$(GREEN)Building gateway...$(NC)"
	go build -o bin/gateway ./cmd/gateway

# Run services
run-auth:
	@echo "$(GREEN)Running auth-service...$(NC)"
	go run ./cmd/auth-service

run-user:
	@echo "$(GREEN)Running user-service...$(NC)"
	go run ./cmd/user-service

run-billing:
	@echo "$(GREEN)Running billing-service...$(NC)"
	go run ./cmd/billing-service

run-notify:
	@echo "$(GREEN)Running notification-service...$(NC)"
	go run ./cmd/notification-service

run-gateway:
	@echo "$(GREEN)Running gateway...$(NC)"
	go run ./cmd/gateway

# Docker commands
docker-build:
	@echo "$(GREEN)Building Docker images...$(NC)"
	docker-compose build

docker-up:
	@echo "$(GREEN)Starting services...$(NC)"
	docker-compose up -d

docker-down:
	@echo "$(GREEN)Stopping services...$(NC)"
	docker-compose down

# Database migrations
migrate-up:
	@echo "$(GREEN)Running all migrations...$(NC)"
	@for service in auth user billing notification; do \
		echo "Migrating $$service..."; \
		migrate -path migrations/$$service -database "$(DATABASE_URL)" up; \
	done

migrate-down:
	@echo "$(GREEN)Rolling back all migrations...$(NC)"
	@for service in auth user billing notification; do \
		echo "Rolling back $$service..."; \
		migrate -path migrations/$$service -database "$(DATABASE_URL)" down 1; \
	done

migrate-auth:
	migrate -path migrations/auth -database "$(AUTH_DATABASE_URL)" up

migrate-user:
	migrate -path migrations/user -database "$(USER_DATABASE_URL)" up

migrate-billing:
	migrate -path migrations/billing -database "$(BILLING_DATABASE_URL)" up

migrate-notification:
	migrate -path migrations/notification -database "$(NOTIFICATION_DATABASE_URL)" up

# Testing
test:
	@echo "$(GREEN)Running tests...$(NC)"
	go test -v ./...

test-coverage:
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# Linting
lint:
	@echo "$(GREEN)Running linter...$(NC)"
	golangci-lint run ./...

# Clean build artifacts
clean:
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf bin/
	rm -rf $(GEN_DIR)/
	rm -f coverage.out coverage.html
