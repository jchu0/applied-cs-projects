#!/bin/bash

# Generate RSA key pair for JWT signing

set -e

KEYS_DIR="${1:-services/auth-service/keys}"

mkdir -p "$KEYS_DIR"

echo "Generating RSA key pair..."

# Generate private key
openssl genrsa -out "$KEYS_DIR/private.pem" 2048

# Generate public key from private key
openssl rsa -in "$KEYS_DIR/private.pem" -pubout -out "$KEYS_DIR/public.pem"

echo "Keys generated successfully in $KEYS_DIR"
echo "  - private.pem (keep secret!)"
echo "  - public.pem (can be distributed)"
