package llm

import (
	"rag-course/config"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// Message represents one entry in a chat conversation.
type Message struct {
	Role    string `json:"role"`    // The sender's role: "system", "user", or "assistant"
	Content string `json:"content"` // Text payload of the message
}

type Client struct {
	cfg config.Config
	sdk openai.Client
}

func New(cfg config.Config) *Client {
	opts := []option.RequestOption{}

	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}

	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}

	return &Client{cfg: cfg, sdk: openai.NewClient(opts...)}
}
