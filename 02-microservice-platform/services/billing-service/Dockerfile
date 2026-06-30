# Build stage
FROM golang:1.21-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o billing-service ./cmd/main.go

# Runtime stage
FROM alpine:3.18

RUN apk --no-cache add ca-certificates tzdata

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/billing-service .
COPY --from=builder /app/migrations ./migrations

# Create non-root user
RUN adduser -D -s /bin/sh appuser
USER appuser

EXPOSE 8080 9090

ENTRYPOINT ["./billing-service"]
