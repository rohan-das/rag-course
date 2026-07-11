package rag

import (
	"fmt"
	"rag-course/vector"
	"strings"
)

const contextPreamble = `Use the following excerpts from the document collection to answer the question.
Cite sources by filename when you draw from them. If the excerpts do not address the question, say so
before answering from general knowledge.`

const unknownSource = "(unknown source)"

// formatContext converts vector search results into a formatted context string
// suitable for LLM prompt injection. Returns an empty string if no results are found.
//
// Example Output:
//
//	Use the following excerpts from the document collection to answer the question.
//	Cite sources by filename when you draw from them. If the excerpts do not address the question, say so
//	before answering from general knowledge.
//
//	--- Excerpts ---
//
//	[1] Source: internal_policy.pdf (similarity 0.89)
//	Employees are eligible for up to 25 days of paid time off per calendar year.
//
//	[2] Source: (unknown source) (similarity 0.76)
//	Remote work requests must be submitted at least two weeks in advance.
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
