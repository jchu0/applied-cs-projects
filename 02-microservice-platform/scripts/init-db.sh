#!/bin/bash

# Initialize databases with migrations

set -e

echo "Waiting for databases to be ready..."
sleep 5

# Run user service migrations
echo "Running user-service migrations..."
PGPASSWORD=userservice_pass psql -h localhost -p 5432 -U userservice -d users -f services/user-service/migrations/001_initial_schema.up.sql

# Run auth service migrations
echo "Running auth-service migrations..."
PGPASSWORD=authservice_pass psql -h localhost -p 5433 -U authservice -d auth -f services/auth-service/migrations/001_initial_schema.up.sql

echo "Migrations completed successfully!"
