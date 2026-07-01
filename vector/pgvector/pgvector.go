package pgvector

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

// Options configures a Store.
type Options struct {
	// DSN is a libpq-style connection string,
	// e.g. "postgres://user:pass@host:5432/db?sslmode=disable".
	DSN string

	// EmbeddingDim is the column width for the embedding vector.
	// Must match the exact dimension produced by the configured embedder (e.g. 1536 for OpenAI).
	EmbeddingDim int
}

// Store is the Postgres + pgvector implementation of vector.Store.
type Store struct {
	pool *pgxpool.Pool
}

// New connects to Postgres, registers the pgvector type with every
// pooled connection, and ensures the schema exists. It returns an
// error if the connection fails, the extension cannot be created, or
// the migration cannot run.
func New(ctx context.Context, opts Options) (*Store, error) {
	if opts.DSN == "" {
		return nil, errors.New("pgvector: DSN is required")
	}
	if opts.EmbeddingDim <= 0 {
		return nil, errors.New("pgvector: EmbeddingDim must be > 0")
	}

	// 1. We read the DSN connection string and build a comprehensive pool configuration.
	// We use pgxpool.ParseConfig because we ultimately want to build a long-lived,
	// managed bundle of connections (a pool) rather than just a single raw connection.
	cfg, err := pgxpool.ParseConfig(opts.DSN)
	if err != nil {
		return nil, fmt.Errorf("parse DSN: %w", err)
	}

	// 2. THE CHICKEN-AND-EGG FIX:
	// We call ensureExtension BEFORE spinning up the pool.
	// This uses a quick, temporary raw connection to tell the Postgres DB to install
	// the vector plugin. We must do this first because the pool's setup phase (AfterConnect below)
	// expects the extension to already be active in the database. If the extension doesn't exist yet,
	// every connection the pool tries to make will fail instantly and crash your application startup.
	if err := ensureExtension(ctx, opts.DSN); err != nil {
		return nil, fmt.Errorf("install extension: %w", err)
	}

	// 3. THE HANDSHAKE REGISTRATION (The Callback):
	// Every single time the pool manager creates a brand-new connection socket in the background,
	// it registers itself with this callback function called AfterConnect.
	// Although we registered the extension to the database beforehand for it to understand the
	// vector datatype, by running `pgxvec.RegisterTypes` inside this hook, we tell the individual
	// connection how to talk about vectors and do stuff with vectors. Without this, the Go driver
	// won't understand how to read or write the data, and would just return it as a raw, messy text string.
	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		return pgxvec.RegisterTypes(ctx, conn)
	}

	// 4. Now that the extension is inside the DB and the type-translation rules are defined,
	// we safely spin up the actual connection pool manager.
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}

	s := &Store{pool: pool}

	// 5. Build our tables and indexes.
	// The pool will internally handle checking out a connection, executing the queries,
	// and returning the connection back to the pool automatically.
	if err := s.migrate(ctx, opts.EmbeddingDim); err != nil {
		pool.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}

	return s, nil
}

// ensureExtension installs the pgvector extension using a single
// throwaway connection that does not have RegisterTypes attached. This
// is the bootstrap step that lets the main pool's AfterConnect succeed
// on a fresh database.
func ensureExtension(ctx context.Context, dsn string) error {
	// We use the raw pgx.Connect approach here because this is a one-off admin task.
	// It doesn't look for any existing pool connections; it just raw-dogs a single TCP
	// connection to run one command, and then we manually destroy it via defer conn.Close().
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return err
	}
	defer conn.Close(ctx) // Manually tearing down the temporary connection right away

	// Register the extension into the DB engine itself so Postgres accepts the 'vector' type
	_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	return err
}

// migrate runs the idempotent schema setup. Each statement is safe to
// re-run, so this can execute safely on every single application startup.
// The CREATE EXTENSION step was already handled by ensureExtension before the pool opened.
func (s *Store) migrate(ctx context.Context, dim int) error {
	stmts := []string{
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS documents (
        id   TEXT PRIMARY KEY,
        content  TEXT NOT NULL,
        metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
        embedding  vector(%d) NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now())
        `, dim),
		`CREATE INDEX IF NOT EXISTS documents_embedding_idx
           ON documents USING hnsw (embedding vector_cosine_ops)`,
	}

	for _, q := range stmts {
		// Unlike pgx.Connect, using s.pool.Exec automatically borrows a pre-warmed connection,
		// runs the query, and checks it back into the pool behind the scenes without breaking it.
		if _, err := s.pool.Exec(ctx, q); err != nil {
			return fmt.Errorf("exec %q: %w", firstLine(q), err)
		}
	}

	return nil
}

func firstLine(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			return s[:i]
		}
	}
	return s
}
