package vector

// Document represents a piece of content to be stored in the vector store.
type Document struct {
	// ID is a stable string identifier rather than a numeric ID.
	ID string

	// Content is the textual content of the chunk that gets embedded.
	// In production, it might be excluded to save space, but it is kept here to
	// review rows in the database easily or to reconstruct the original documents.
	Content string

	// Metadata contains arbitrary structured data associated with a chunk
	// (e.g., source file name, page number, or ingestion timestamp).
	Metadata map[string]string

	// Embedding holds the high-dimensional float32 vector produced by an LLM embedder.
	Embedding []float32
}

// Result represents a single hit or match returned from a similarity query.
type Result struct {
	Document

	// Score is the similarity between the query vector and the stored vector.
	// Based on the backend's index configuration (e.g., pgVector's cosine similarity),
	// it will fall in the range of 0 to 1 for normalized vectors.
	// A score > 0.8 is strongly related; < 0.4 is likely just noise.
	Score float32
}

// Store defines the contract for our vector store backend. Using an interface
// allows us to easily swap Postgres (pgVector) out for Weaviate or any other
// storage mechanism without altering application logic.
type Store interface {
	// Upsert inserts new documents or updates/replaces existing ones by their ID.
	Upsert(ctx context.context, docs []Document) error

	// Query performs a nearest neighbor search and returns the topK documents
	// most similar to the provided search embedding.
	Query(ctx context.context, embedding []float32, topK int) ([]Result, error)

	// Delete removes one or more documents from the store using their explicit string IDs.
	Delete(ctx context.context, ids []string) error

	// DeleteBySource removes every document whose metadata source field matches
	// the provided source string (useful for batch-deleting tagged data).
	DeleteBySource(ctx context.context, source string) error

	// Close releases underlying database pools, network connections, and resources
	// to prevent resource leaks.
	Close() error
}
