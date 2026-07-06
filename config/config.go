package config

import (
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

// Config holds the configuration settings for the LLM client.
type Config struct {
	// BaseURL points at any OpenAI-compatible chat-completions
	// server: api.openai.com, a local Ollama at :11434/v1, LM
	// Studio, Groq, Together, vLLM, and so on. The wire protocol is
	// the same; only the URL and model name change.
	BaseURL string

	// APIKey is sent as `Authorization: Bearer <key>` when non-empty.
	// Local servers usually accept any value (or none); hosted
	// providers require their own key.
	APIKey string

	// Model is the chat-completions model identifier. Defaults to
	// gpt-4o-mini so a fresh OpenAI key works with no further setup.
	Model string

	// SystemPromptFile is the path to a text/markdown file whose
	// contents become the conversation's system message. A missing
	// file is silently treated as "no system prompt".
	SystemPromptFile string

	// DatabaseURL is the libpq-style DSN for the
	// Postgres + pgvector instance. Empty means "no vector store" —
	// chat still runs, just without retrieval. Populated from
	// DATABASE_URL.
	DatabaseURL string

	// EmbeddingDim is the dimensionality of the
	// embedding model that will populate the vector column. It is
	// baked into the column type at first migration (vector(1536) is
	// a different SQL type from vector(768)) and cannot be changed
	// afterward without recreating the table.
	//
	//   text-embedding-3-small  → 1536
	//   text-embedding-3-large  → 3072
	//   nomic-embed-text         → 768
	EmbeddingDim int

	// embedder endpoint config. EmbeddingBaseURL and
	// EmbeddingAPIKey let the embedder talk to a different OpenAI-
	// compatible endpoint than the chat client. The motivating case:
	// a hosted chat model (Ollama Cloud, OpenAI, Groq, ...) plus a
	// local embedder (Ollama, LM Studio, ...) — some hosted providers
	// do not expose embedding models. When EmbeddingBaseURL is empty,
	// the embedder reuses BaseURL/APIKey, preserving "one server for
	// everything" for the simple case.
	EmbeddingBaseURL string
	EmbeddingAPIKey  string
	EmbeddingModel   string
	IngestDir        string
	ProcessedDir     string
}

// Load initializes the Config struct by reading environment variables.
// It sets sensible defaults if optional fields are missing.
func Load() Config {
	// A missing .env is not an error — fall back to the real process
	// environment so the program runs without one.
	_ = godotenv.Load()

	cfg := Config{
		BaseURL:          os.Getenv("OPENAI_BASE_URL"),
		APIKey:           os.Getenv("OPENAI_API_KEY"),
		Model:            os.Getenv("OPENAI_MODEL"),
		SystemPromptFile: os.Getenv("SYSTEM_PROMPT_FILE"),
		DatabaseURL:      os.Getenv("DATABASE_URL"),
		EmbeddingDim:     atoiOr(os.Getenv("EMBEDDING_DIM"), 0),
		EmbeddingBaseURL: os.Getenv("EMBEDDING_BASE_URL"),
		EmbeddingAPIKey:  os.Getenv("EMBEDDING_API_KEY"),
		EmbeddingModel:   os.Getenv("EMBEDDING_MODEL"),
		IngestDir:        os.Getenv("INGEST_DIR"),
		ProcessedDir:     os.Getenv("PROCESSED_DIR"),
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}

	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
	}

	// NOTE: The default configuration uses 768 dimensions to maintain compatibility
	// with the Nomic embedding model (Ollama's default) and the pre-existing
	// 'documents.embedding' vector column in the database.
	//
	// If you switch to an OpenAI embedding model, you must update this value:
	// - Use 1536 for 'text-embedding-3-small'
	// - Use 3072 for 'text-embedding-3-large'
	//
	// WARNING: The vector dimension size is strictly defined when the database
	// table is first created. If you change your embedding model later, the
	// database will reject the new vectors. To switch models in the future,
	// you must drop the 'documents' table and recreate it with the new dimension size.
	if cfg.EmbeddingDim == 0 {
		cfg.EmbeddingDim = 768
	}

	// When the user hasn't pointed the embedder at a
	// separate endpoint, reuse the chat endpoint and key — preserving
	// "one OpenAI-compatible server for everything" for the simple
	// case. When EMBEDDING_BASE_URL IS set we leave the API key alone:
	// a different host means a different (or no) credential, and
	// silently borrowing the chat key would send it to a server that
	// didn't ask for it.
	if cfg.EmbeddingBaseURL == "" {
		cfg.EmbeddingBaseURL = cfg.BaseURL
		if cfg.EmbeddingAPIKey == "" {
			cfg.EmbeddingAPIKey = cfg.APIKey
		}
	}

	if cfg.EmbeddingModel == "" {
		cfg.EmbeddingModel = "nomic-embed-text"
	}
	if cfg.IngestDir == "" {
		cfg.IngestDir = "./documents"
	}
	if cfg.ProcessedDir == "" {
		cfg.ProcessedDir = "./documents/processed"
	}

	return cfg
}

// atoiOr parses s as an int, returning fallback
// when s is empty or invalid. Used so an unset EMBEDDING_DIM means
// "apply default" rather than zero.
func atoiOr(s string, fallback int) int {
	if s == "" {
		return fallback
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return fallback
	}
	return n
}
