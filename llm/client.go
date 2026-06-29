package llm

import (
	"context"
	"rag-course/config"
	"strings"

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

func (c *Client) ChatStream(ctx context.Context, messages []Message, onDelta func(string)) (Message, error) {
	stream := c.sdk.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Model:    c.cfg.Model,
		Messages: toSDKMessages(messages),
	})
	defer stream.Close()

	var content strings.Builder
	role := "assistant"

	for stream.Next() {
		// A chunk represents a single, complete data packet received from the API stream at a specific moment in time.
		chunk := stream.Current()
		if len(chunk.Choices) == 0 {
			// The Choices field is an array (or slice) inside the chunk.
			// OpenAI allows you to request multiple alternative responses simultaneously (by setting the n parameter in the API request).
			// Each item in this array is one of those alternative responses.
			continue
		}

		delta := chunk.Choices[0].Delta
		// In a standard, non-streaming API call, this field is called Message and contains the entire response. 
		// However, in a streaming API call, it is called a Delta because it represents only the change or difference from the previous chunk.
		if delta.Role != "" {
			// The API populates the role explicitly on the initial stream chunk to optimize bandwidth.
			role = string(delta.Role)
		}

		if delta.Content != "" {
			// Only process chunks that contain actual text content.
			content.WriteString(delta.Content)
			if onDelta != nil {
				onDelta(delta.Content)
			}
		}
	}

	if err := stream.Err(); err != nil {
		return Message{}, err
	}

	return Message{Role: role, Content: content.String()}, nil
}

func toSDKMessages(messages []Message) []openai.ChatCompletionMessageParamUnion {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, m := range messages {
		switch m.Role {
		case "system":
			out = append(out, openai.SystemMessage(m.Content))
		case "assistant":
			out = append(out, openai.AssistantMessage(m.Content))
		default:
			out = append(out, openai.UserMessage(m.Content))
		}
	}

	return out
}
