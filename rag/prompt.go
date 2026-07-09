package rag

import (
	"fmt"
	"rag-course/vector"
	"strings"
)

const contextPreamble = `Use the following excerpts from the document collection to answer the question.
Cite sources by filename when you draw from them. If the excerpts do not address the question, say so
before answer from general knowledge.`

const unknownSource = "(unknown source)"

// formatContext serializes a list of vector search results into a formatted string
// for LLM context injection. Returns an empty string if no hits are provided.
func formatContext(hits []vector.Result) string {
	if len(hits) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString(contextPreamble)
	sb.WriteString("\n\n--- Excerpts ---\n\n")

	for i, h := range hits {
		source := h.Metadata["source"]
		if source == "" {
			// Fallback for edge cases where vectors are manually inserted without source metadata
			source = unknownSource
		}
		fmt.Fprintf(&sb, "[%d] Source: %s (similarity %.2f)\n%s\n\n", i+1, source, h.Score, h.Content)
	}
	return strings.TrimSpace(sb.String())
}
