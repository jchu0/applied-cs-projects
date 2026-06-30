#!/bin/bash
set -e

# Kong Admin API URL
KONG_ADMIN_URL="${KONG_ADMIN_URL:-http://localhost:8001}"

echo "Waiting for Kong to be ready..."
until curl -s "${KONG_ADMIN_URL}/status" > /dev/null 2>&1; do
    sleep 2
done
echo "Kong is ready!"

echo "Configuring Kong API Gateway..."

# ============================================================================
# Services
# ============================================================================

echo "Creating services..."

# User Service
curl -s -X POST "${KONG_ADMIN_URL}/services" \
    -d "name=user-service" \
    -d "url=http://user-service:8080" \
    -d "retries=3" \
    -d "connect_timeout=10000" \
    -d "read_timeout=60000" \
    -d "write_timeout=60000" | jq .

# Auth Service
curl -s -X POST "${KONG_ADMIN_URL}/services" \
    -d "name=auth-service" \
    -d "url=http://auth-service:8080" \
    -d "retries=3" \
    -d "connect_timeout=10000" \
    -d "read_timeout=60000" \
    -d "write_timeout=60000" | jq .

# Billing Service
curl -s -X POST "${KONG_ADMIN_URL}/services" \
    -d "name=billing-service" \
    -d "url=http://billing-service:8080" \
    -d "retries=3" \
    -d "connect_timeout=10000" \
    -d "read_timeout=60000" \
    -d "write_timeout=60000" | jq .

# Notification Service
curl -s -X POST "${KONG_ADMIN_URL}/services" \
    -d "name=notification-service" \
    -d "url=http://notification-service:8080" \
    -d "retries=3" \
    -d "connect_timeout=10000" \
    -d "read_timeout=60000" \
    -d "write_timeout=60000" | jq .

# ============================================================================
# Routes
# ============================================================================

echo "Creating routes..."

# User Service Routes
curl -s -X POST "${KONG_ADMIN_URL}/services/user-service/routes" \
    -d "name=user-api" \
    -d "paths[]=/api/v1/users" \
    -d "strip_path=false" \
    -d "preserve_host=false" | jq .

# Auth Service Routes
curl -s -X POST "${KONG_ADMIN_URL}/services/auth-service/routes" \
    -d "name=auth-api" \
    -d "paths[]=/api/v1/auth" \
    -d "strip_path=false" \
    -d "preserve_host=false" | jq .

# Billing Service Routes
curl -s -X POST "${KONG_ADMIN_URL}/services/billing-service/routes" \
    -d "name=billing-api" \
    -d "paths[]=/api/v1/billing" \
    -d "strip_path=false" \
    -d "preserve_host=false" | jq .

# Billing Webhooks (Stripe) - No auth required
curl -s -X POST "${KONG_ADMIN_URL}/services/billing-service/routes" \
    -d "name=billing-webhooks" \
    -d "paths[]=/api/v1/webhooks/stripe" \
    -d "strip_path=false" \
    -d "preserve_host=false" | jq .

# Notification Service Routes
curl -s -X POST "${KONG_ADMIN_URL}/services/notification-service/routes" \
    -d "name=notification-api" \
    -d "paths[]=/api/v1/notifications" \
    -d "strip_path=false" \
    -d "preserve_host=false" | jq .

# Health Check Routes (no auth)
curl -s -X POST "${KONG_ADMIN_URL}/services/user-service/routes" \
    -d "name=user-health" \
    -d "paths[]=/health/user" \
    -d "strip_path=true" \
    -d "preserve_host=false" | jq .

curl -s -X POST "${KONG_ADMIN_URL}/services/auth-service/routes" \
    -d "name=auth-health" \
    -d "paths[]=/health/auth" \
    -d "strip_path=true" \
    -d "preserve_host=false" | jq .

curl -s -X POST "${KONG_ADMIN_URL}/services/billing-service/routes" \
    -d "name=billing-health" \
    -d "paths[]=/health/billing" \
    -d "strip_path=true" \
    -d "preserve_host=false" | jq .

curl -s -X POST "${KONG_ADMIN_URL}/services/notification-service/routes" \
    -d "name=notification-health" \
    -d "paths[]=/health/notification" \
    -d "strip_path=true" \
    -d "preserve_host=false" | jq .

# ============================================================================
# Global Plugins
# ============================================================================

echo "Configuring global plugins..."

# CORS Plugin (Global)
curl -s -X POST "${KONG_ADMIN_URL}/plugins" \
    -d "name=cors" \
    -d "config.origins[]=*" \
    -d "config.methods[]=GET" \
    -d "config.methods[]=POST" \
    -d "config.methods[]=PUT" \
    -d "config.methods[]=PATCH" \
    -d "config.methods[]=DELETE" \
    -d "config.methods[]=OPTIONS" \
    -d "config.headers[]=Authorization" \
    -d "config.headers[]=Content-Type" \
    -d "config.headers[]=X-Tenant-ID" \
    -d "config.exposed_headers[]=X-Request-ID" \
    -d "config.credentials=true" \
    -d "config.max_age=3600" | jq .

# Request Transformer - Add X-Request-ID
curl -s -X POST "${KONG_ADMIN_URL}/plugins" \
    -d "name=correlation-id" \
    -d "config.header_name=X-Request-ID" \
    -d "config.generator=uuid#counter" \
    -d "config.echo_downstream=true" | jq .

# File Log Plugin (Global)
curl -s -X POST "${KONG_ADMIN_URL}/plugins" \
    -d "name=file-log" \
    -d "config.path=/dev/stdout" \
    -d "config.reopen=false" | jq .

# Prometheus Plugin (Global)
curl -s -X POST "${KONG_ADMIN_URL}/plugins" \
    -d "name=prometheus" \
    -d "config.per_consumer=true" \
    -d "config.status_code_metrics=true" \
    -d "config.latency_metrics=true" \
    -d "config.bandwidth_metrics=true" \
    -d "config.upstream_health_metrics=true" | jq .

# ============================================================================
# Rate Limiting
# ============================================================================

echo "Configuring rate limiting..."

# Global Rate Limiting (fallback)
curl -s -X POST "${KONG_ADMIN_URL}/plugins" \
    -d "name=rate-limiting" \
    -d "config.minute=100" \
    -d "config.hour=1000" \
    -d "config.policy=local" \
    -d "config.fault_tolerant=true" \
    -d "config.hide_client_headers=false" | jq .

# Stricter rate limiting for auth endpoints
curl -s -X POST "${KONG_ADMIN_URL}/routes/auth-api/plugins" \
    -d "name=rate-limiting" \
    -d "config.minute=20" \
    -d "config.hour=100" \
    -d "config.policy=local" \
    -d "config.fault_tolerant=true" | jq .

# ============================================================================
# JWT Authentication
# ============================================================================

echo "Configuring JWT authentication..."

# Create JWT plugin for protected routes
# User API (protected)
curl -s -X POST "${KONG_ADMIN_URL}/routes/user-api/plugins" \
    -d "name=jwt" \
    -d "config.key_claim_name=iss" \
    -d "config.claims_to_verify[]=exp" \
    -d "config.run_on_preflight=false" | jq .

# Billing API (protected)
curl -s -X POST "${KONG_ADMIN_URL}/routes/billing-api/plugins" \
    -d "name=jwt" \
    -d "config.key_claim_name=iss" \
    -d "config.claims_to_verify[]=exp" \
    -d "config.run_on_preflight=false" | jq .

# Notification API (protected)
curl -s -X POST "${KONG_ADMIN_URL}/routes/notification-api/plugins" \
    -d "name=jwt" \
    -d "config.key_claim_name=iss" \
    -d "config.claims_to_verify[]=exp" \
    -d "config.run_on_preflight=false" | jq .

# Create JWT consumer for the platform
curl -s -X POST "${KONG_ADMIN_URL}/consumers" \
    -d "username=microservices-platform" | jq .

# Note: You need to add the JWT credential with the public key
# This should be done after services are running and keys are generated
echo ""
echo "============================================================================"
echo "JWT Setup Note:"
echo "After generating RSA keys, add the JWT credential with:"
echo ""
echo "curl -X POST ${KONG_ADMIN_URL}/consumers/microservices-platform/jwt \\"
echo "    -F 'key=microservices-platform' \\"
echo "    -F 'algorithm=RS256' \\"
echo "    -F 'rsa_public_key=@keys/public.pem'"
echo "============================================================================"

# ============================================================================
# Request Size Limiting
# ============================================================================

echo "Configuring request size limits..."

# Global request size limit
curl -s -X POST "${KONG_ADMIN_URL}/plugins" \
    -d "name=request-size-limiting" \
    -d "config.allowed_payload_size=10" \
    -d "config.size_unit=megabytes" | jq .

# ============================================================================
# IP Restriction (Optional - for admin routes)
# ============================================================================

# Uncomment to restrict admin access to specific IPs
# curl -s -X POST "${KONG_ADMIN_URL}/routes/admin-api/plugins" \
#     -d "name=ip-restriction" \
#     -d "config.allow[]=127.0.0.1" \
#     -d "config.allow[]=10.0.0.0/8" | jq .

# ============================================================================
# Bot Detection
# ============================================================================

echo "Configuring bot detection..."

curl -s -X POST "${KONG_ADMIN_URL}/plugins" \
    -d "name=bot-detection" \
    -d "config.deny[]=" | jq .

# ============================================================================
# Response Transformer
# ============================================================================

echo "Configuring response headers..."

curl -s -X POST "${KONG_ADMIN_URL}/plugins" \
    -d "name=response-transformer" \
    -d "config.add.headers[]=X-Kong-Proxy:true" \
    -d "config.add.headers[]=X-Content-Type-Options:nosniff" \
    -d "config.add.headers[]=X-Frame-Options:DENY" \
    -d "config.add.headers[]=X-XSS-Protection:1; mode=block" | jq .

echo ""
echo "============================================================================"
echo "Kong configuration complete!"
echo ""
echo "Gateway URL: http://localhost:8000"
echo "Admin API:   http://localhost:8001"
echo "Admin GUI:   http://localhost:8002"
echo ""
echo "Available endpoints:"
echo "  - GET  /api/v1/users/*          -> user-service"
echo "  - POST /api/v1/auth/*           -> auth-service"
echo "  - GET  /api/v1/billing/*        -> billing-service"
echo "  - POST /api/v1/webhooks/stripe  -> billing-service (no auth)"
echo "  - GET  /api/v1/notifications/*  -> notification-service"
echo "  - GET  /health/*                -> health checks (no auth)"
echo "============================================================================"
