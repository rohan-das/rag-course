package config

import (
	"os"

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
}

// Load initializes the Config struct by reading environment variables.
// It sets sensible defaults if optional fields are missing.
func Load() Config {
	_ = godotenv.Load()

	cfg := Config{
		BaseURL:          os.Getenv("OPENAI_BASE_URL"),
		APIKey:           os.Getenv("OPENAI_API_KEY"),
		Model:            os.Getenv("OPENAI_MODEL"),
		SystemPromptFile: os.Getenv("SYSTEM_PROMPT_FILE"),
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}

	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
	}

	return cfg
}
