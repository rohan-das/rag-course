package rag

import (
	"context"
	"fmt"
	"rag-course/llm"
	"rag-course/vector"
)

const defaultTopK = 5

type Options struct {
	// Maximum number of document chunks to retrieve.
	TopK int

	// Optional query rewriter used before retrieval.
	Rewriter *Rewriter
}

type Retriever struct {
	embedder llm.Embedder
	store    vector.Store
	rewriter *Rewriter
	topK     int
}

func New(embedder llm.Embedder, store vector.Store, opts Options) *Retriever {
	topK := opts.TopK
	if topK <= 0 {
		topK = defaultTopK
	}
	return &Retriever{
		embedder: embedder,
		store:    store,
		rewriter: opts.Rewriter,
		topK:     topK,
	}
}

/*

[User Message]
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ 1. RETRIEVER.GO (The Project Manager)                 │
│    • Receives the messy chat history.                  │
│    • Calls the Rewriter to clean up the request.       │
└───────────────────────┬────────────────────────────────┘
                        │ (Sends entire history)
                        ▼
┌────────────────────────────────────────────────────────┐
│ 2. REWRITER.GO (The Research Assistant)                │
│    • Looks at the history (e.g., "Who created it?").   │
│    • Uses a small AI to turn it into crisp keywords.  │
│    • Returns: "Go language creators"                   │
└───────────────────────┬────────────────────────────────┘
                        │ (Returns clean keywords)
                        ▼
┌────────────────────────────────────────────────────────┐
│ 3. BACK TO RETRIEVER.GO                                │
│    • Converts "Go language creators" into vectors.     │
│    • Queries the Vector DB for matching document hits. │
└───────────────────────┬────────────────────────────────┘
                        │ (Sends raw database hits)
                        ▼
┌────────────────────────────────────────────────────────┐
│ 4. PROMPT.GO (The Presentation Designer)               │
│    • Takes the raw database document snippets.         │
│    • Massages them into a clean, readable text block.  │
│    • Returns: Structured context string.               │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
       [Final structured prompt ready for the Main AI]

*/

// Retrieve is the main coordinator. It manages the whole process from start to finish:
// Instead of searching the database with the entire conversation history, it extracts only
// the single, last relevant intent. It looks at the chat log and uses a rewriter to turn
// messy human chat shortcuts—like "Who made it?", where the meaning of "it" was clarified
// earlier in the conversation history—into a meaningful, specific, and singular search query
// (like "Go language creators"). Once it has this clear query, it turns those words into
// numbers (a vector) so the database can understand the meaning, pulls out the most helpful
// matching documents, and neatly arranges them into a clean text block.
func (r *Retriever) Retrieve(ctx context.Context, history []llm.Message) (string, error) {
	// 1. Get the optimized search text (either an AI-rewritten query or the raw last message).
	query := r.buildQuery(ctx, history)
	if query == "" {
		return "", nil // Abort early if there is no active user message to search for.
	}

	// 2. Convert the text query into a vector
	vecs, err := r.embedder.Embed(ctx, []string{query})
	if err != nil {
		return "", fmt.Errorf("embed query: %w", err)
	}
	if len(vecs) == 0 {
		return "", nil
	}

	// 3. Search the vector database using the calculated vector to find the
	// most relevant document chunks (up to the configured topK limit).
	hits, err := r.store.Query(ctx, vecs[0], r.topK)
	if err != nil {
		return "", fmt.Errorf("vector query: %w", err)
	}
	if len(hits) == 0 {
		return "", nil
	}

	// 4. Return the formatted text
	return formatContext(hits), nil
}

// buildQuery constructs the text query that will be sent to the document
// retrieval system.
//
// The latest user message is not always a good search query by itself.
// People naturally use conversational language with pronouns and omitted
// context:
//
//	User: "I'm learning Go."
//	Assistant: "It's a highly concurrent language."
//	User: "Who created it?"
//
// A retriever or vector database sees only the literal query. Searching for
// "Who created it?" provides almost no useful signal because the important
// subject ("Go") is missing.
//
// To improve retrieval quality, we optionally pass the entire conversation
// history to a query rewriter. The rewriter's job is to infer the missing
// context and produce a standalone, search-oriented query. For example:
//
//	"Who created it?"  -> "Go programming language creators"
//
// This rewritten query is optimized for retrieval rather than conversation.
// It should preserve the user's intent while making implicit references
// explicit, improving the chances of retrieving relevant documents.
//
// The rewriter is optional. It may be disabled entirely (r.rewriter == nil)
// to avoid the additional latency and cost of making another LLM request for
// every search. Even when enabled, the rewrite operation can fail due to
// network problems, service outages, timeouts, rate limits, or invalid
// responses.
//
// Regardless of why rewriting is unavailable, retrieval should continue.
// Rather than returning an error and preventing the search from happening,
// we gracefully fall back to the user's most recent message. While this
// query may produce less accurate search results, it keeps the retrieval
// pipeline operational and avoids turning a recoverable rewrite failure
// into a user-visible error.
func (r *Retriever) buildQuery(ctx context.Context, history []llm.Message) string {
	if r.rewriter != nil {
		if q, err := r.rewriter.Rewrite(ctx, history); err == nil && q != "" {
			return q
		}
	}
	return lastUserMessage(history)
}

// lastUserMessage extracts the most recent message sent by the user from the
// conversation history.
//
// A retrieval system needs a focused search query, not the entire conversation
// transcript. Passing the full chat history would introduce irrelevant context
// and make it harder for the vector database to identify the user's actual
// information need.
//
// This function isolates the latest user request, which represents the current
// search intent. For example:
//
//	[1] User: "What is Go?"
//	[2] Assistant: "A language by Google."
//	[3] User: "Who created it?" <--- Current search request.
//
// Walking backward through the history allows us to quickly find the active
// user message while ignoring previous turns that have already been handled.
func lastUserMessage(history []llm.Message) string {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "user" {
			return history[i].Content
		}
	}
	return ""
}
