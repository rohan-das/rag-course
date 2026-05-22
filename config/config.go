package config

import (
	"os"

	"github.com/joho/godotenv"
)

// Config holds the configuration settings for the LLM client.
type Config struct {
	BaseURL          string // BaseURL is the endpoint for the OpenAI-compatible API server.
	APIKey           string // APIKey is the authentication token; may be a placeholder for local providers.
	Model            string // Model specifies the identifier of the LLM to be used.
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
