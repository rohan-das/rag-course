package llm

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http"

	"github.com/openai/openai-go/v3"
)

const captionPrompt = "Describe this image in 2-3 sentences for a search index. Focus on the visible subject - what it is, key details, style. Do not interpret or speculate; describe only what is shown."

func (c *Client) HasVision() bool {
	return c.cfg.VisionModel != ""
}

func (c *Client) DescribeImage(ctx context.Context, mime string, image []byte) (string, error) {
	if c.cfg.VisionModel == "" {
		return "", errors.New("no vision model configured")
	}

	if len(image) == 0 {
		return "", errors.New("empty image")
	}

	if mime == "" {
		// Examples of automatically detected MIME types:
		// - "image/png"
		// - "image/jpeg"
		// - "image/webp"
		// - "image/gif"
		mime = http.DetectContentType(image)
	}

	// Example resulting dataURL string:
	// "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4..."
	dataURL := "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(image)

	resp, err := c.sdk.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: c.cfg.VisionModel,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
				openai.TextContentPart(captionPrompt),
				openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
					// URL accepts both data URLs (e.g., "data:image/png;base64,...")
					// and public HTTPS URLs (e.g., "https://example.com/photo.jpg").
					URL: dataURL,
				}),
			}),
		},
	})
	if err != nil {
		return "", err
	}
	return resp.Choices[0].Message.Content, nil
}
