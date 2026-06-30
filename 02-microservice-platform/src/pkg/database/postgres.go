package database

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mlai/microservice-platform/pkg/config"
)

// PostgresDB wraps a PostgreSQL connection pool
type PostgresDB struct {
	pool *pgxpool.Pool
}

// NewPostgresDB creates a new PostgreSQL connection pool
func NewPostgresDB(ctx context.Context, cfg *config.DatabaseConfig) (*PostgresDB, error) {
	poolConfig, err := pgxpool.ParseConfig(cfg.DSN())
	if err != nil {
		return nil, fmt.Errorf("failed to parse database config: %w", err)
	}

	// Configure pool settings
	poolConfig.MaxConns = int32(cfg.MaxConns)
	poolConfig.MinConns = int32(cfg.MinConns)
	poolConfig.MaxConnLifetime = cfg.MaxConnLife
	poolConfig.MaxConnIdleTime = cfg.MaxConnIdle

	// Create pool
	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}

	// Test connection
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &PostgresDB{pool: pool}, nil
}

// Pool returns the underlying connection pool
func (db *PostgresDB) Pool() *pgxpool.Pool {
	return db.pool
}

// Close closes the connection pool
func (db *PostgresDB) Close() {
	db.pool.Close()
}

// Health checks the database health
func (db *PostgresDB) Health(ctx context.Context) error {
	return db.pool.Ping(ctx)
}

// SetTenantContext sets the tenant context for row-level security
func (db *PostgresDB) SetTenantContext(ctx context.Context, tenantID string) error {
	_, err := db.pool.Exec(ctx, "SET app.current_tenant = $1", tenantID)
	return err
}

// WithTenant returns a connection with tenant context set
func (db *PostgresDB) WithTenant(ctx context.Context, tenantID string) (*pgxpool.Conn, error) {
	conn, err := db.pool.Acquire(ctx)
	if err != nil {
		return nil, err
	}

	_, err = conn.Exec(ctx, "SET app.current_tenant = $1", tenantID)
	if err != nil {
		conn.Release()
		return nil, fmt.Errorf("failed to set tenant context: %w", err)
	}

	return conn, nil
}

// Transaction represents a database transaction
type Transaction struct {
	tx pgxpool.Tx
}

// Begin starts a new transaction
func (db *PostgresDB) Begin(ctx context.Context) (*Transaction, error) {
	tx, err := db.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	return &Transaction{tx: tx}, nil
}

// Commit commits the transaction
func (t *Transaction) Commit(ctx context.Context) error {
	return t.tx.Commit(ctx)
}

// Rollback rolls back the transaction
func (t *Transaction) Rollback(ctx context.Context) error {
	return t.tx.Rollback(ctx)
}

// Tx returns the underlying transaction
func (t *Transaction) Tx() pgxpool.Tx {
	return t.tx
}
