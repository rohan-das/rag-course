package app

import (
	"context"
	"log"
	"os"
	"rag-course/chat"
	"rag-course/config"
	"rag-course/llm"
	"rag-course/vector"
	"rag-course/vector/pgvector"
)

func Run(ctx context.Context, cfg config.Config) error {
	// A stderr-tagged logger so connection-related
	// status lines ("vector store ready", "vector store disabled: ...")
	// are clearly distinguishable from chat output on stdout.
	logger := log.New(os.Stderr, "[rag] ", log.LstdFlags)

	client := llm.New(cfg)

	// Open the vector store. A nil store with a
	// nil error means "no DATABASE_URL configured" — the chat path
	// works fine without a database, so we surface the reason and
	// keep going. Any real error (bad DSN, server unreachable,
	// migration failure) is also logged but not fatal.
	store, err := openStore(ctx, cfg)
	if err != nil {
		logger.Printf("vector store disabled: %v", err)
	}

	// Defer Close so the connection pool drains
	// cleanly on exit (Ctrl-C, REPL quit, or any error). Guarded by
	// the nil-check because openStore returns a nil interface when
	// DATABASE_URL is unset.
	if store != nil {
		defer store.Close()
		logger.Printf("vector store ready")
	}

	return chat.RunREPL(ctx, client, chat.Options{
		SystemPromptFile: cfg.SystemPromptFile,
	})
}

// openStore returns a configured vector.Store, or
// (nil, nil) when no DATABASE_URL is set. The (nil, nil) case is
// intentional and signals "feature disabled, not an error" to the
// caller.
//
// Note: we explicitly drop the concrete *pgvector.Store on error and
// return a nil interface. Returning the typed nil directly would box a
// nil pointer into a non-nil vector.Store interface, defeating the
// "if store != nil" check in Run.
func openStore(ctx context.Context, cfg config.Config) (vector.Store, error) {
	if cfg.DatabaseURL == "" {
		return nil, nil
	}
	s, err := pgvector.New(ctx, pgvector.Options{
		DSN:          cfg.DatabaseURL,
		EmbeddingDim: cfg.EmbeddingDim,
	})
	if err != nil {
		return nil, err
	}
	return s, nil
}
