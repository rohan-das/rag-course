package web

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"rag-course/ingest"
	"rag-course/llm"
	"rag-course/rag"
	"rag-course/vector"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

//go:embed templates/*.gohtml
var templatesFS embed.FS

const maxUploadBytes = 10 << 20

type uploadResponse struct {
	Source string `json:"source"`
	Bytes  int    `json:"bytes"`
	Chunks int    `json:"chunks"`
}

type Options struct {
	Addr             string
	SystemPromptFile string
	Title            string
	Store            vector.Store
	ProcessedDir     string
	ImagesDir        string
}

type Server struct {
	client       *llm.Client
	embedder     *llm.Client
	retriever    *rag.Retriever
	store        vector.Store
	processedDir string
	imagesDir    string
	tpl          *template.Template
	system       string
	title        string
}

func (s *Server) Routes() http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.Recoverer)
	r.Get("/chat", s.handleChatPage)
	r.Post("/api/chat/stream", s.handleChatStream)
	r.Post("/api/upload", s.handleUpload)

	return r
}

type chatRequest struct {
	Messages []llm.Message `json:"messages"`
}

func (s *Server) handleUpload(w http.ResponseWriter, r *http.Request) {
	if s.store == nil {
		http.Error(w, "ingest is not configured (no vector store)", http.StatusServiceUnavailable)
		return
	}

	// Wrap the request body with a size limit. Any reads beyond
	// maxUploadBytes will fail, preventing excessively large uploads.
	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)

	// Parse the multipart/form-data request. Uploaded files larger than
	// maxUploadBytes or malformed multipart bodies return an error.
	if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
		http.Error(w, "upload too large or malformed: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Retrieve the uploaded file from the multipart form field named "file".
	// header contains metadata such as the original filename.
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "missing 'file' field: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Use only the base filename to avoid directory traversal issues
	// (e.g., "/data/employee.txt" becomes "employee.txt").
	name := filepath.Base(header.Filename)
	if !ingest.IsSupported(name) {
		http.Error(w, "unsupported format", http.StatusUnsupportedMediaType)
		return
	}

	// Read the uploaded file into memory.
	content, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, "read upload: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Process the document, generate embeddings, and store them.
	chunks, err := ingest.ProcessContent(r.Context(), name, content, ingest.Options{}, s.embedder, s.store)
	if err != nil {
		log.Printf("[web] upload ingest failed for %q: %v", name, err)
		http.Error(w, "ingest failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Archive the successfully processed file.
	if s.processedDir != "" {
		dest := filepath.Join(s.processedDir, name)
		if err := os.MkdirAll(s.processedDir, 0o755); err != nil {
			log.Printf("[web] mkdir %s: %v", s.processedDir, err)
		} else if err := os.WriteFile(dest, content, 0o644); err != nil {
			log.Printf("[web] archive %s: %v", dest, err)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(uploadResponse{
		Source: name,
		Bytes:  len(content),
		Chunks: chunks,
	})
}

func (s *Server) handleChatStream(w http.ResponseWriter, r *http.Request) {
	// Streaming responses require the ResponseWriter to implement
	// http.Flusher. Flush() forces any buffered data to be sent to
	// the client immediately, allowing the client to receive partial
	// responses (tokens/chunks) as they are generated.
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json:"+err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Messages) == 0 {
		http.Error(w, "messages must not be empty", http.StatusBadRequest)
		return
	}

	if last := req.Messages[len(req.Messages)-1]; last.Role != "user" {
		http.Error(w, "last message must be from user", http.StatusBadRequest)
		return
	}

	history := req.Messages
	if s.system != "" {
		history = append([]llm.Message{{Role: "system", Content: s.system}}, history...)
	}

	turn := history
	if s.retriever != nil {
		ctxText, err := s.retriever.Retrieve(r.Context(), history)
		if err != nil {
			log.Printf("[web] retrieval error: %v", err)
		} else {
			turn = withInlineContext(history, ctxText)
		}
	}

	// Configure the response as a Server-Sent Events (SSE) stream.
	//
	// Content-Type: text/event-stream
	//   Indicates that the response will be an SSE stream.
	//
	// Cache-Control: no-cache
	//   Prevents browsers and proxies from caching streamed events.
	//
	// Connection: keep-alive
	//   Keeps the HTTP connection open so events can be sent
	//   continuously instead of returning a single response.
	//
	// X-Accel-Buffering: no
	//   Disables buffering in reverse proxies such as Nginx so that
	//   each event reaches the client immediately.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	// Send the headers immediately so the client knows the stream
	// has started before the first event is produced.
	flusher.Flush()

	send := func(event, data string) {
		if event != "" {
			fmt.Fprintf(w, "event: %s\n", event)
		}
		fmt.Fprintf(w, "data: %s\n\n", data)

		// Flush each event immediately instead of waiting for the
		// response buffer to fill up.
		flusher.Flush()
	}

	_, err := s.client.ChatStream(r.Context(), turn, func(delta string) {
		enc, _ := json.Marshal(delta)
		send("delta", string(enc))
	})
	if err != nil {
		enc, _ := json.Marshal(err.Error())
		send("error", string(enc))
		return
	}
	send("done", `""`)
}

func withInlineContext(history []llm.Message, contextText string) []llm.Message {
	if len(history) == 0 || contextText == "" {
		return history
	}
	last := history[len(history)-1]
	if last.Role != "user" {
		return history
	}
	out := make([]llm.Message, len(history))
	copy(out, history)
	out[len(out)-1] = llm.Message{
		Role:    "user",
		Content: contextText + "\n\n--- Question ---\n\n" + last.Content,
	}
	return out
}

func (s *Server) handleChatPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := s.tpl.ExecuteTemplate(w, "chat.gohtml", map[string]any{
		"Title": s.title,
	}); err != nil {
		log.Printf("[web] template error: %v", err)
	}
}

// Run starts the HTTP server and manages its entire lifecycle concurrently.
func (s *Server) Run(ctx context.Context, addr string) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           s.Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	// 1. ERROR CHANNEL + SERVER GOROUTINE
	//
	// Buffered (capacity = 1) so sending the first error never blocks simply
	// because the receiver hasn't reached the select yet.
	errCh := make(chan error, 1)

	// ListenAndServe() blocks forever while serving requests, so run it in a
	// goroutine. This allows the main goroutine to simultaneously wait for
	// either a shutdown signal or an unexpected server failure.
	go func() {
		err := srv.ListenAndServe()

		// Shutdown() intentionally causes ListenAndServe() to return
		// http.ErrServerClosed. That's the expected exit path, not a failure.
		if !errors.Is(err, http.ErrServerClosed) {
			// Example: port already in use, permission denied, etc.
			errCh <- err
		}

		// SAFETY NET: If the server closed normally via Shutdown(), we push 'nil' down the channel
		// to explicitly signal that the background process ended cleanly without failure.
		errCh <- nil
	}()

	// 2. ORCHESTRATION
	//
	// Wait for whichever happens first:
	//   1. External shutdown (context canceled)
	//   2. Internal server failure
	select {

	// SCENARIO A: External shutdown
	//
	// Triggered when ctx is canceled (Ctrl+C, SIGTERM, Kubernetes shutdown, etc.).
	case <-ctx.Done():

		// The parent context is already canceled, so it cannot be reused.
		// Create a fresh context with a 5-second deadline to allow a graceful shutdown.
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		// Shutdown():
		//   1. Stops accepting new requests.
		//   2. Waits for active requests to finish.
		//   3. Forces shutdown if they exceed the timeout.
		_ = srv.Shutdown(shutdownCtx)

		// Graceful exit.
		return nil

	// SCENARIO B: Server failure
	//
	// Example: startup failure because the port is already in use.
	case err := <-errCh:

		// Return the original error so the caller knows why the server stopped.
		return err
	}
}

func New(client, embedder *llm.Client, retriever *rag.Retriever, opts Options) (*Server, error) {
	tpl, err := template.ParseFS(templatesFS, "templates/*.gohtml")
	if err != nil {
		return nil, fmt.Errorf("parse templates: %w", err)
	}

	title := opts.Title
	if title == "" {
		title = "RAG Chat"
	}

	return &Server{
		client:       client,
		embedder:     embedder,
		retriever:    retriever,
		store:        opts.Store,
		processedDir: opts.ProcessedDir,
		imagesDir:    opts.ImagesDir,
		tpl:          tpl,
		system:       readSystemPrompt(opts.SystemPromptFile),
		title:        title,
	}, nil
}

func readSystemPrompt(path string) string {
	if path == "" {
		return ""
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return ""
	}

	return strings.TrimSpace(string(data))
}
