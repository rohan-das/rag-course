package web

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"mime"
	"mime/multipart"
	"net/http"
	"rag-course/ingest"
	"regexp"
)

// The following code implements an HTTP middleware for Prompt Injection
// and Guardrail Defense designed specifically for a Retrieval-Augmented Generation (RAG) system.

// In a RAG application, prompt injection can happen in two ways:
// Direct Injection: A user sends malicious text directly in a chat endpoint.
// Indirect Injection: A user uploads a document (e.g., PDF, TXT) containing malicious text
// designed to hijack the Large Language Model (LLM) once ingested into the vector database and retrieved into context.

// This code acts as a safety barrier in front of your HTTP API routes to inspect both incoming JSON payloads
// and uploaded multipart files before they reach your LLM pipeline.

const (
	injectionScanBudget = 64 * 1024 // 64 KB limit for regex scanning, regex can be expensive
	maxJSONBodyBytes    = 1 << 20   // 1 MB limit for JSON bodies
	maxMultipartBytes   = 10 << 20  // 10 MB limit for file uploads
)

// injectionPatterns contains a list of regular expressions (compiled with (?i) for case-insensitivity)
// targeting common jailbreaks and prompt injections:
//   - Instruction Overrides: "ignore previous instructions", "disregard all prior system prompts".
//   - Context Erasure: "forget everything...".
//   - Role Takeovers / Personas: "you are now DAN", "act as an AI with no restrictions".
//   - Developer Mode / Jailbreaks: "do anything now", "developer mode enabled".
//   - Prompt Leakage Attempts: "reveal your system prompt", "show me verbatim above".
//   - Safety Bypass: "override your guardrails", "turn off content policy".
//   - Role-Tag Smuggling: Attacks attempting to trick special prompt formatting mechanisms using tokens
//     like <|im_start|>, <|system|>, or [INST].
var injectionPatterns = []*regexp.Regexp{
	// "ignore (all|any|the) (previous|prior|above|earlier) (instructions|prompts|messages)"
	regexp.MustCompile(`(?i)\bignore\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier|prior\s+system)\s+(?:instructions?|prompts?|messages?|directives?|rules?)`),
	// "disregard ..."
	regexp.MustCompile(`(?i)\bdisregard\s+(?:all\s+|any\s+|the\s+|your\s+|previous\s+|prior\s+|above\s+|system\s+)+(?:instructions?|prompts?|messages?|directives?|rules?)`),
	// "forget everything ..."
	regexp.MustCompile(`(?i)\bforget\s+(?:all\s+|everything\s+|prior\s+|previous\s+|your\s+)+(?:instructions?|prompts?|messages|context|rules?|above)`),
	// Role-takeover: "you are now X", "act as Y", "pretend to be Z"
	regexp.MustCompile(`(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as)\s+(?:dan|jailbreak|developer\s+mode|unrestricted|an?\s+ai\s+with\s+no\s+(?:restrictions|filters|rules))`),
	// "Do Anything Now" / "DAN mode" / "developer mode enabled"
	regexp.MustCompile(`(?i)\b(?:do\s+anything\s+now|dan\s+mode(?:\s+enabled)?|developer\s+mode\s+(?:enabled|activated|on))\b`),
	// Reveal-the-prompt attacks
	regexp.MustCompile(`(?i)\breveal\s+(?:your\s+|the\s+)?(?:system\s+)?(?:prompt|instructions|directives)`),
	regexp.MustCompile(`(?i)\b(?:what|show|tell\s+me)\s+(?:are|were|is|me)?\s*(?:your|the)?\s*(?:system\s+)?(?:prompt|instructions|directives)\b`),
	regexp.MustCompile(`(?i)\b(?:print|output|repeat|echo|show)\s+(?:everything\s+|all\s+(?:of\s+)?|me\s+|verbatim\s+)?(?:above|prior|previous|your\s+(?:system\s+)?(?:prompt|instructions))`),
	// Safety-bypass language
	regexp.MustCompile(`(?i)\b(?:override|bypass|circumvent|disable|turn\s+off)\s+(?:your\s+|the\s+|all\s+)?(?:safety|guardrails|restrictions|filters|content\s+policy|safeguards)`),
	// ChatML / role-tag smuggling. Local templates that honor these
	// tokens will treat the embedded role as authoritative.
	regexp.MustCompile(`(?i)<\s*\|?\s*im_start\s*\|?\s*>`),
	regexp.MustCompile(`(?i)<\s*\|?\s*(?:system|assistant|developer)\s*\|?\s*>`),
	regexp.MustCompile(`(?i)\[\s*(?:INST|SYSTEM|/SYSTEM)\s*\]`),
}

/*
Incoming Request (JSON or Multipart Upload)
                 │
                 ▼
       [ InjectionDefense ]
                 │
  ┌──────────────┴──────────────┐
  ▼                             ▼
JSON Payload?           Multipart Upload?
(Chat endpoints)        (Document ingestion)
  │                             │
  ├─ Limiter: 1MB               ├─ Limiter: 10MB
  └─ Inspect `user` messages     ├─ Inspect form text values
                                └─ Scan file headers/content
                 │
                 ▼
     [ Regular Expression Match? ]
         │                  │
        YES                 NO
         │                  │
         ▼                  ▼
  Reject Request      Pass to Next Handler
 (HTTP 400 Error)       (`next.ServeHTTP`)
*/

// scanForInjection checks if a string contains any prompt injection patterns.
func scanForInjection(s string) string {
	// Limits scanning to the first 64KB of text/files to avoid performance degradation
	// or Denial of Service (DoS) regex attacks.
	if len(s) > injectionScanBudget {
		s = s[:injectionScanBudget] // Truncate to 64 KB budget
	}

	for _, p := range injectionPatterns {
		if p.MatchString(s) {
			// Returns the exact regex string that triggered the match.
			// Example value: "(?i)\\b(?:you\\s+are\\s+now|act\\s+as|pretend\\s+to\\s+be)..."
			return p.String()
		}
	}

	return ""
}

// InjectionDefense is an HTTP middleware that intercepts requests to scan
// both direct chat inputs (JSON) and indirect document uploads (multipart) for prompt injections.
func InjectionDefense(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mt, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
		switch mt {
		case "application/json":
			inspectJSON(w, r, next)
		case "multipart/form-data":
			inspectMultiPart(w, r, next)
		default:
			next.ServeHTTP(w, r)
		}
	})
}

func inspectJSON(w http.ResponseWriter, r *http.Request, next http.Handler) {
	// Read the request body up to a maximum limit of 1 MB (maxJSONBodyBytes)
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxJSONBodyBytes))
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}

	var req chatRequest
	if jsonErr := json.Unmarshal(body, &req); jsonErr == nil {
		// Scan only user-authored messages, ignoring system or assistant messages
		for _, m := range req.Messages {
			if m.Role != "user" {
				continue
			}
			// Run the injection scan against the message content
			if hit := scanForInjection(m.Content); hit != "" {
				log.Printf("[web-defense] blocked chat request: pattern=%q route=%s", hit, r.URL.Path)
				http.Error(w, "request rejected by injection-defense filter", http.StatusBadRequest)
				return
			}
		}
	}

	// Note: r.Body is a single-pass stream. Reading it above consumed all bytes.
	// We recreate an io.ReadCloser from the saved byte slice using bytes.NewReader
	// and io.NopCloser so downstream handlers (like the main chat API) can re-read the body.
	r.Body = io.NopCloser(bytes.NewReader(body))
	next.ServeHTTP(w, r)
}

// inspectMultiPart processes multipart/form-data requests (such as image or document uploads).
func inspectMultiPart(w http.ResponseWriter, r *http.Request, next http.Handler) {
	// Restrict total request body size to 10 MB (maxMultipartBytes)
	r.Body = http.MaxBytesReader(w, r.Body, maxMultipartBytes)

	// Parse form fields and file headers into memory/temp files
	if err := r.ParseMultipartForm(maxMultipartBytes); err != nil {
		// If parsing fails, fall back and pass through to the next handler
		next.ServeHTTP(w, r)
		return
	}

	// Step 1: Scan plain-text form parameters for direct prompt injection
	for name, vals := range r.MultipartForm.Value {
		for _, v := range vals {
			if hit := scanForInjection(v); hit != "" {
				log.Printf("[web-defense] blocked chat request: pattern=%q field=%q route=%s", hit, name, r.URL.Path)
				http.Error(w, "request rejected by injection-defense filter", http.StatusBadRequest)
				return
			}
		}
	}

	// Step 2: Scan uploaded document files for indirect prompt injection
	for field, files := range r.MultipartForm.File {
		for _, fh := range files {
			// Skip unsupported file formats
			if !ingest.IsSupported(fh.Filename) {
				continue
			}

			// Scan the file content
			hit, err := scanFilePart(fh)
			if err != nil {
				log.Printf("[web-defense] read upload %q: %v", fh.Filename, err)
				http.Error(w, "uploaded document rejected by injection-defense filter", http.StatusBadRequest)
				return
			}
			if hit != "" {
				log.Printf("[web-defense] blocked upload: pattern=%q file=%q field=%q route=%q",
					hit, fh.Filename, field, r.URL.Path)
				http.Error(w, "request rejected by injection-defense filter", http.StatusBadRequest)
				return
			}
		}
	}

	next.ServeHTTP(w, r)
}

// scanFilePart opens an uploaded file header and reads its prefix up to the injection budget.
func scanFilePart(fh *multipart.FileHeader) (string, error) {
	f, err := fh.Open()
	if err != nil {
		return "", err
	}
	defer f.Close()

	// Safely read up to 64 KB + 1 byte without reading the entire file into memory
	buf, err := io.ReadAll(io.LimitReader(f, injectionScanBudget+1))
	if err != nil {
		return "", err
	}

	// Convert bytes to string and scan
	return scanForInjection(string(buf)), nil
}
