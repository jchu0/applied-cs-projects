package common

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// Database wraps a PostgreSQL connection pool
type Database struct {
	Pool *pgxpool.Pool
}

// NewDatabase creates a new database connection pool
func NewDatabase(ctx context.Context, databaseURL string) (*Database, error) {
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse database URL: %w", err)
	}

	// Connection pool settings
	config.MaxConns = 25
	config.MinConns = 5
	config.MaxConnLifetime = time.Hour
	config.MaxConnIdleTime = 30 * time.Minute
	config.HealthCheckPeriod = time.Minute

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}

	// Verify connection
	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &Database{Pool: pool}, nil
}

// Close closes the database connection pool
func (db *Database) Close() {
	db.Pool.Close()
}

// SetTenantContext sets the current tenant for Row-Level Security
func (db *Database) SetTenantContext(ctx context.Context, tenantID string) error {
	_, err := db.Pool.Exec(ctx, "SET app.current_tenant = $1", tenantID)
	if err != nil {
		return fmt.Errorf("failed to set tenant context: %w", err)
	}
	return nil
}

// WithTenant returns a connection with tenant context set
func (db *Database) WithTenant(ctx context.Context, tenantID string) (*pgxpool.Conn, error) {
	conn, err := db.Pool.Acquire(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to acquire connection: %w", err)
	}

	_, err = conn.Exec(ctx, "SET app.current_tenant = $1", tenantID)
	if err != nil {
		conn.Release()
		return nil, fmt.Errorf("failed to set tenant context: %w", err)
	}

	return conn, nil
}

// HealthCheck verifies database connectivity
func (db *Database) HealthCheck(ctx context.Context) error {
	return db.Pool.Ping(ctx)
}
