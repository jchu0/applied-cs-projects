# Microservice Platform

A production-ready microservice platform with Auth, User, Billing, and Notification services.

## Architecture

```
┌─────────────────┐
│   API Gateway   │
└────────┬────────┘
         │
┌────────┼────────┐────────────────┐
│        │        │                │
▼        ▼        ▼                ▼
Auth     User     Billing     Notification
Service  Service  Service     Service
│        │        │                │
▼        ▼        ▼                ▼
Redis    PostgreSQL PostgreSQL   PostgreSQL
         │        │                │
         └────────┴────────────────┘
                  │
              NATS (Events)
```

## Services

| Service | gRPC Port | HTTP Port | Description |
|---------|-----------|-----------|-------------|
| User Service | 9090 | 8090 | User management, roles, permissions |
| Auth Service | 9091 | 8091 | Authentication, JWT tokens, sessions |
| Billing Service | 9092 | 8092 | Subscriptions, payments (Stripe) |
| Notification Service | 9093 | 8093 | Email, SMS, push notifications |

## Quick Start

### Prerequisites

- Go 1.21+
- Docker & Docker Compose
- Make
- protoc (Protocol Buffers compiler)

### Setup

1. **Start infrastructure services:**

```bash
docker-compose up -d postgres-users redis nats jaeger prometheus grafana
```

2. **Run database migrations:**

```bash
# Install golang-migrate
go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest

# Run migrations
export DATABASE_URL="postgres://postgres:postgres@localhost:5432/users?sslmode=disable"
make migrate-user
```

3. **Generate protobuf code:**

```bash
make proto-deps  # Install protoc plugins
make proto       # Generate Go code
```

4. **Build and run services:**

```bash
# Build all services
make build

# Run user service
make run-user

# Run auth service (in another terminal)
make run-auth
```

### Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Stop all services
docker-compose down
```

## Development

### Project Structure

```
src/
├── cmd/                    # Service entry points
│   ├── auth-service/
│   ├── user-service/
│   ├── billing-service/
│   └── notification-service/
├── internal/               # Private service code
│   ├── auth/
│   ├── user/
│   ├── billing/
│   └── notification/
├── pkg/                    # Shared packages
│   ├── config/
│   ├── database/
│   ├── logging/
│   ├── middleware/
│   └── pb/                 # Generated protobuf code
├── proto/                  # Protobuf definitions
├── migrations/             # Database migrations
├── configs/                # Configuration files
└── deployments/            # Docker files
```

### Configuration

Services can be configured via:
- YAML config files (see `configs/`)
- Environment variables (prefix with service name)

Example environment variables:
```bash
SERVICE_GRPC_PORT=9090
DATABASE_HOST=localhost
DATABASE_PASSWORD=secret
REDIS_HOST=redis
```

### Testing

```bash
# Run all tests
make test

# Run tests with coverage
make test-coverage
```

### gRPC APIs

Use [grpcurl](https://github.com/fullstorydev/grpcurl) or a gRPC client:

```bash
# List services
grpcurl -plaintext localhost:9090 list

# Create user
grpcurl -plaintext -d '{
  "tenant_id": "00000000-0000-0000-0000-000000000001",
  "email": "test@example.com",
  "password": "password123",
  "first_name": "Test",
  "last_name": "User"
}' localhost:9090 user.v1.UserService/CreateUser

# Login
grpcurl -plaintext -d '{
  "email": "test@example.com",
  "password": "password123",
  "tenant_id": "00000000-0000-0000-0000-000000000001"
}' localhost:9091 auth.v1.AuthService/Login
```

## Observability

- **Jaeger UI**: http://localhost:16686 (distributed tracing)
- **Prometheus**: http://localhost:9090 (metrics)
- **Grafana**: http://localhost:3000 (dashboards, admin/admin)

## Technology Stack

- **Language**: Go 1.21
- **RPC**: gRPC with Protocol Buffers
- **Database**: PostgreSQL 15 with Row-Level Security
- **Cache/Sessions**: Redis 7
- **Message Broker**: NATS JetStream
- **Tracing**: OpenTelemetry + Jaeger
- **Metrics**: Prometheus
- **Logging**: Zap (structured JSON)

## Security Features

- JWT authentication with RS256 signing
- Multi-tenant data isolation (Row-Level Security)
- Password hashing with bcrypt
- Session management with Redis
- Request rate limiting

## License

MIT
