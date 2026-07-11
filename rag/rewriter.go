package rag

import (
	"context"
	"fmt"
	"rag-course/llm"
	"strings"
)

const rewriteSystemPrompt = `You rewrite the user's latest message into a standalone search query.

Given the conversation, output a single search query that:
- Captures the topic and intent of the latest user message.
- Resolves pronouns and references using prior turns ("it", "they", "that one").
- Stays concise - keywords and short phrases, not full sentences.

If the latest user message already stands on its own with no references to prior turns, output it verbatim.

Output only the query. No preamble, no quotes, no explanation.`

type Rewriter struct {
	client *llm.Client
}

func NewRewriter(client *llm.Client) *Rewriter {
	return &Rewriter{client: client}
}

// Rewrite converts the current conversational request into a standalone query
// suitable for document retrieval.
//
// Human conversations rely heavily on context. Users naturally ask follow-up
// questions using pronouns or omitted subjects:
//
//	User: "I'm learning Go."
//	Assistant: "It's designed for concurrency."
//	User: "Who created it?"
//
// While a human immediately understands that "it" refers to the Go programming
// language, a search engine or vector database generally does not. Searching
// for "Who created it?" provides very little semantic information because the
// primary subject is missing.
//
// Rewrite solves this problem by analyzing the entire conversation history and
// generating a concise query that preserves the user's intent while making all
// implied references explicit. For the example above, the rewritten query might
// become:
//
//	"Go programming language creators" or "Who created go?"
//
// The rewritten text is intended for retrieval rather than display. It serves
// as the canonical search query that will later be embedded and matched against
// indexed documents.
//
// Rewriting is intentionally lightweight. It focuses on resolving conversational
// references and preserving intent instead of answering the user's question or
// introducing new information.
func (r *Rewriter) Rewrite(ctx context.Context, history []llm.Message) (string, error) {
	// Extract the latest user message. This represents the request that needs
	// to be rewritten. If there is no user input, there is nothing to process.
	last := lastUserMessage(history)
	if last == "" {
		return "", nil
	}

	// If the conversation contains no assistant responses yet, there is usually
	// no conversational context to resolve. Running an LLM rewrite would add
	// latency and cost while producing nearly the same text, so return the
	// user's original message unchanged.
	if !hasAssistantTurn(history) {
		return last, nil
	}

	// Send the conversation transcript to the rewriting model. The system prompt
	// instructs the model to rewrite the latest request into a standalone,
	// retrieval-friendly query, while the formatted conversation provides the
	// context needed to resolve pronouns and omitted references.
	msgs := []llm.Message{
		{Role: "system", Content: rewriteSystemPrompt},
		{Role: "user", Content: formatConversation(history)},
	}

	// Generate the rewritten query. Streaming isn't necessary here because the
	// result is consumed internally by the retrieval pipeline rather than shown
	// incrementally to the user. We reuse ChatStream because the client does
	// not currently expose a non-streaming chat API.
	reply, err := r.client.ChatStream(ctx, msgs, nil)
	if err != nil {
		return "", fmt.Errorf("rewrite call: %w", err)
	}

	// Normalize the model's output. Models occasionally include surrounding
	// whitespace or quote characters, which are formatting artifacts rather than
	// part of the intended query.
	out := strings.TrimSpace(reply.Content)
	out = strings.Trim(out, `"'`)

	// Guard against unexpected model output. If rewriting succeeds but produces
	// an empty string, fall back to the user's original message so retrieval can
	// continue instead of failing.
	if out == "" {
		return last, nil
	}

	return out, nil
}

// hasAssistantTurn reports whether the conversation contains an assistant reply.
func hasAssistantTurn(history []llm.Message) bool {
	for _, m := range history {
		if m.Role == "assistant" {
			return true
		}
	}
	return false
}

// formatConversation converts the chat history into a readable transcript for
// the rewriting model.
//
// Each message is prefixed with its speaker so the model can distinguish
// between user requests and assistant responses. The transcript ends with an
// explicit instruction telling the model to rewrite only the latest user
// message into a standalone search query.
func formatConversation(history []llm.Message) string {
	var sb strings.Builder
	sb.WriteString("Conversation so far:\n\n")
	for _, m := range history {
		switch m.Role {
		case "user":
			sb.WriteString("User: ")
		case "assistant":
			sb.WriteString("Assistant: ")
		default:
			continue
		}
		sb.WriteString(m.Content)
		sb.WriteString("\n\n")
	}
	// Reinforce the expected output. The system prompt already defines this task,
	// but repeating it here helps keep the model focused on the required format.
	sb.WriteString("Rewrite the user's latest message as a standalone search query.")
	return sb.String()
}
