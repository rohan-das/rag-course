package web

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"html/template"
	"io/fs"
	"log"
	"net/http"
	"os"
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

	return r
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
