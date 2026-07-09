package ingest

import "strings"

// chunk splits text into overlapping chunks of about `size` bytes.
//
// Instead of cutting exactly at `size`, it scans backward near the end
// of the window to find the last suitable break point.
//
//  1. Paragraph break ("\n\n")
//  2. Sentence end (". ")
//  3. Space (" ")
//
// This keeps sentences and paragraphs together whenever possible.
//
// This matters because each chunk is turned into a single embedding.
// If a sentence is split in half, both chunks lose some context.
// Keeping complete thoughts together usually produces better embeddings
// and improves retrieval quality.
//
// The chunk size is only approximate. If a suitable boundary is found
// near the end of the window, the chunk is shortened slightly instead
// of cutting in the middle of a sentence or word.
//
// This implementation is intentionally simple. More advanced chunkers
// exist (token-based or semantic chunkers), but this approach is
// accurate enough for our use case and easy to understand.
//
// The function uses byte indexes instead of rune indexes because the
// boundaries we search for ("\n\n", ". ", and " ") are all ASCII.
// For our input, this is a simple and practical approach.
func chunk(text string, size, overlap int) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	if len(text) <= size {
		return []string{text}
	}

	// Prevent invalid overlap values.
	// If overlap >= size, start would never move forward.
	if overlap < 0 {
		overlap = 0
	}

	if overlap >= size {
		overlap = size / 2
	}

	// Only use a boundary if it appears in the last 30% of the window.
	// Otherwise we'd create lots of small chunks instead of chunks that
	// are close to the requested size.
	threshold := size * 7 / 10

	var chunks []string
	n := len(text)
	start := 0

	for start < n {
		end := start + size

		// Last chunk: take whatever is left.
		if end >= n {
			if part := strings.TrimSpace(text[start:]); part != "" {
				chunks = append(chunks, part)
			}
			break
		}

		window := text[start:end]

		// Prefer to split at a paragraph, then a sentence, then a word,
		// as long as the boundary is close to the end of the window.
		switch {
		case strings.LastIndex(window, "\n\n") >= threshold:
			end = start + strings.LastIndex(window, "\n\n") + 2
		case strings.LastIndex(window, ". ") >= threshold:
			end = start + strings.LastIndex(window, ". ") + 2
		case strings.LastIndex(window, " ") >= threshold:
			end = start + strings.LastIndex(window, " ") + 1
		}

		if part := strings.TrimSpace(text[start:end]); part != "" {
			chunks = append(chunks, part)
		}

		// Move to the next chunk while keeping some overlap so that
		// information near the boundary appears in both chunks.
		start = end - overlap
	}

	return chunks
}
