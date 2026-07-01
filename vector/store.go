package vector

import "context"

// Document represents a piece of content to be stored in the vector store.
type Document struct {
	// ID is a stable unique identifier used for upserts and deletions.
	// Example format: "<source-path>#<chunk-index>"
	ID string

	// Content is the textual content of the chunk that gets embedded.
	// In production, it might be excluded to save space, but it is kept here to
	// review rows in the database easily or to reconstruct the original documents.
	Content string

	// Metadata contains arbitrary structured data associated with a chunk
	// (e.g., source file name, page number, or ingestion timestamp).
	Metadata map[string]string

	// Embedding is the high-dimensional float32 vector produced by an LLM embedder.
	// All documents within the same store must share identical dimensional lengths.
	Embedding []float32
}

// Result represents a single hit or match returned from a similarity query.
type Result struct {
	Document

	// Score represents the similarity metric between the query vector and the document vector.
	// For standard normalized vectors (e.g., OpenAI, Nomic), this falls between [0, 1].
	//
	// Empirical Threshold Reference:
	//   >= 0.80 : Strongly related
	//   0.60 - 0.80 : Related
	//   0.40 - 0.60 : Weakly related
	//   < 0.40 : Likely noise
	Score float32
}

// Store enforces the behavior required by any underlying vector database backend.
// All execution paths accept a context.Context to enforce deadlines, timeouts,
// and graceful cancellations.
type Store interface {
	// Upsert inserts new records or overwrites existing documents sharing the same ID.
	// Implementations should execute this within a single transaction where supported.
	Upsert(ctx context.Context, docs []Document) error

	// Query executes a nearest-neighbor lookup and returns the topK most similar results.
	// Mismatches between input embedding length and store dimensions must return an error.
	Query(ctx context.Context, embedding []float32, topK int) ([]Result, error)

	// Delete removes targeted documents by their unique string IDs.
	// Requests for non-existent IDs must resolve as a no-op rather than an error.
	Delete(ctx context.Context, ids []string) error

	// DeleteBySource purges all chunks bound to a specific source identity.
	// This acts as a sanitation step during re-ingestion to prevent older,
	// trailing chunks from becoming orphaned when a file shrinks.
	DeleteBySource(ctx context.Context, source string) error

	// Close gracefully tears down active database pools and network attachments.
	// Subsequent invocations on an already closed store must return a no-op.
	Close() error
}
