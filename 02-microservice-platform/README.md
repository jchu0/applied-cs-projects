# Microservice Platform

A complete microservice platform with Auth, Billing, Notifications, and User management services.

## Overview

This project implements a production-ready microservice architecture featuring:

- **User Service** - User CRUD operations with multi-tenant support
- **Auth Service** - JWT-based authentication with RS256 signing
- **gRPC Communication** - High-performance inter-service communication
- **Multi-Tenant Architecture** - Row-level security for data isolation
- **Redis Sessions** - Scalable session management

## Architecture

```
┌─────────────────┐
│   API Gateway   │
└────────┬────────┘
         │
    ┌────┼────┐
    │    │    │
┌───▼──┐ ┌▼──┐ ┌▼────┐
│ User │ │Auth│ │Billing│
│  Svc │ │Svc │ │ Svc  │
└──┬───┘ └─┬─┘ └──┬───┘
   │       │      │
┌──▼───────▼──────▼──┐
│     PostgreSQL     │
│       Redis        │
└────────────────────┘
```

## Quick Start

### Prerequisites

- Go 1.21+
- Docker & Docker Compose
- Protocol Buffers compiler (protoc)

### Development Setup

1. **Start infrastructure services:**
   ```bash
   make docker-up
   ```

2. **Run database migrations:**
   ```bash
   make migrate-up
   ```

3. **Run services:**
   ```bash
   # Terminal 1 - User Service
   make run-user

   # Terminal 2 - Auth Service
   make run-auth
   ```

### Configuration

Services are configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GRPC_PORT` | gRPC server port | 9090 |
| `HTTP_PORT` | HTTP server port | 8080 |
| `DATABASE_URL` | PostgreSQL connection string | postgres://localhost/users |
| `REDIS_URL` | Redis connection string | redis://localhost:6379 |
| `JWT_SECRET` | JWT signing secret | (development key) |
| `ENVIRONMENT` | Environment name | development |

## Project Structure

```
02-microservice-platform/
├── cmd/
│   ├── user-service/      # User service entry point
│   └── auth-service/      # Auth service entry point
├── internal/
│   ├── common/            # Shared utilities
│   │   ├── config.go      # Configuration loading
│   │   ├── database.go    # PostgreSQL connection
│   │   └── logger.go      # Structured logging
│   ├── user/              # User service implementation
│   │   ├── repository.go  # Data access layer
│   │   └── service.go     # Business logic
│   └── auth/              # Auth service implementation
│       ├── jwt.go         # JWT token management
│       ├── service.go     # Auth business logic
│       └── session.go     # Redis session store
├── proto/                 # Protocol buffer definitions
│   ├── common/            # Shared types
│   ├── user/              # User service API
│   └── auth/              # Auth service API
├── migrations/            # Database migrations
│   ├── user/              # User DB migrations
│   └── auth/              # Auth DB migrations
├── deployments/           # Docker and deployment configs
│   ├── docker-compose.yml
│   ├── Dockerfile.user
│   └── Dockerfile.auth
├── go.mod
├── Makefile
└── README.md
```

## API Overview

### User Service (gRPC)

```protobuf
service UserService {
    rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
    rpc GetUser(GetUserRequest) returns (GetUserResponse);
    rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
    rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse);
    rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
    rpc GetUserByEmail(GetUserByEmailRequest) returns (GetUserByEmailResponse);
}
```

### Auth Service (gRPC)

```protobuf
service AuthService {
    rpc Login(LoginRequest) returns (LoginResponse);
    rpc Logout(LogoutRequest) returns (LogoutResponse);
    rpc RefreshToken(RefreshTokenRequest) returns (RefreshTokenResponse);
    rpc ValidateToken(ValidateTokenRequest) returns (ValidateTokenResponse);
    rpc Register(RegisterRequest) returns (RegisterResponse);
}
```

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage
```

## Multi-Tenant Support

The platform supports multiple tenants with complete data isolation:

1. **Row-Level Security (RLS)** - PostgreSQL policies enforce tenant isolation
2. **Tenant Context** - Every request includes tenant identification
3. **Composite Keys** - Unique constraints include tenant_id

Example query with tenant isolation:
```sql
-- RLS automatically filters to current tenant
SELECT * FROM users WHERE email = 'user@example.com';
```

## Security Features

- **JWT Authentication** - RS256 signed tokens
- **Password Hashing** - bcrypt with configurable cost
- **Session Management** - Redis-backed with expiry
- **RBAC** - Role-based access control
- **Audit Logging** - Track all auth events

## Development Commands

```bash
# Build all services
make build

# Generate protobuf code
make proto

# Format code
make fmt

# Run linter
make lint

# Clean build artifacts
make clean
```

## Phase 1 Implementation Status

- [x] Project structure and build system
- [x] Protocol buffer schemas
- [x] User Service CRUD operations
- [x] Auth Service (login, JWT)
- [x] PostgreSQL migrations
- [x] Docker Compose setup
- [x] Unit tests

## Next Steps (Phase 2)

- [ ] Complete Auth Service (refresh, logout)
- [ ] Add session management
- [ ] Implement Billing Service
- [ ] Integrate Stripe API
- [ ] Build Notification Service
- [ ] Set up message broker (NATS)
- [ ] Implement event publishing

## License

MIT
