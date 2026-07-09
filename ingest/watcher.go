package ingest

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"rag-course/llm"
	"rag-course/vector"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
)

// The quiet period to wait before processing a file.
// When a file is being copied or written to, the OS triggers multiple write events in rapid succession.
// Waiting 500 milliseconds after the LAST event ensures the file is completely written and closed
// by the operating system before we try to read and process it.
const debounceDelay = 500 * time.Millisecond

func Watcher(ctx context.Context, opts Options, embedder llm.Embedder, store vector.Store, logger *log.Logger) error {
	if filepath.Clean(opts.SourceDir) == filepath.Clean(opts.ProcessedDir) {
		return errors.New("source and processed directories must differ")
	}
	if err := os.MkdirAll(opts.SourceDir, 0o755); err != nil {
		return fmt.Errorf("create source dir: %w", err)
	}
	if err := os.MkdirAll(opts.ProcessedDir, 0o755); err != nil {
		return fmt.Errorf("create processed dir: %w", err)
	}

	w, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("create watcher: %w", err)
	}
	defer w.Close()

	if err := w.Add(opts.SourceDir); err != nil {
		return fmt.Errorf("watch source dir: %w", err)
	}

	// Convert the relative path into a full, absolute system path.
	// e.g. "documents/report.txt" -> "/Users/me/project/documents/report.txt"
	// Note: This relies entirely on the current working directory (where the app is executed from).
	processedAbs, err := filepath.Abs(opts.ProcessedDir)
	if err != nil {
		return fmt.Errorf("resolve processed dir: %w", err)
	}

	handle := func(path string) {
		if err := processOne(ctx, path, opts, embedder, store); err != nil {
			logger.Printf("process %s: %v", filepath.Base(path), err)
			return
		}

		dst := filepath.Join(opts.ProcessedDir, filepath.Base(path))
		if err := os.Rename(path, dst); err != nil {
			logger.Printf("move %s to processed: %v", filepath.Base(path), err)
			return
		}

		logger.Printf("ingested %s", filepath.Base(path))
	}

	entries, err := os.ReadDir(opts.SourceDir)
	if err != nil {
		return fmt.Errorf("read source dir: %w", err)
	}

	for _, e := range entries {
		if e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue // We don't want to process anything that is a directory of a file name starting with .
		}
		go handle(filepath.Join(opts.SourceDir, e.Name()))
	}

	// Handling the case when program is running and the file is dropped into the documents directory
	var (
		timersMu sync.Mutex
		timers   = make(map[string]*time.Timer)
	)

	schedule := func(path string) {
		timersMu.Lock()
		defer timersMu.Unlock()

		// 1. If a timer ALREADY exists for this file, reset it!
		if t, ok := timers[path]; ok {
			t.Reset(debounceDelay)
			return
		}

		// 2. If no timer exists, create a new one that waits 500ms
		timers[path] = time.AfterFunc(debounceDelay, func() {
			timersMu.Lock()
			delete(timers, path) // Clean up the map
			timersMu.Unlock()

			handle(path) // Finally process the file!
		})
	}

	for {
		select {
		case <-ctx.Done():
			return nil
		case ev, ok := <-w.Events:
			if !ok {
				return nil
			}
			// Only care about files being created or written to.
			// Skip other events like CHMOD, REMOVE, or RENAME.
			if ev.Op&(fsnotify.Create|fsnotify.Write) == 0 {
				continue
			}

			// ev.Name contains the path relative to the watched directory.
			// e.g., If watching "documents", ev.Name might be "documents/report.txt"
			// If watching "/var/data", ev.Name might be "/var/data/report.txt"
			if !shouldProcess(ev.Name, processedAbs) {
				continue
			}

			// Pass the file path (e.g., "documents/report.txt") to the debounced timer handler
			schedule(ev.Name)
		case err, ok := <-w.Errors:
			if !ok {
				return nil
			}
			logger.Printf("watcher error: %v", err)
		}
	}
}

func shouldProcess(path, processedAbs string) bool {
	// e.g. path: "documents/report.txt"
	// filepath.Base(path) : "report.txt"
	// Skip hidden files or OS temporary files (like ".DS_Store" or "._report.txt")
	if strings.HasPrefix(filepath.Base(path), ".") {
		return false
	}

	// Get metadata about the file/folder at this path.
	// Skip if the file was deleted before we could read it (err != nil) or if it's a directory.
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return false
	}

	// Convert the relative path into a full, absolute system path.
	// e.g. "documents/report.txt" -> "/Users/me/project/documents/report.txt"
	abs, err := filepath.Abs(path)
	if err != nil {
		return false
	}

	// Anti-Infinite Loop Shield:
	// If the absolute path of this file starts with the "processed" directory path plus a separator,
	// it means this file is inside the processed folder. Skip it so we don't re-ingest moved files.
	// e.g. If processedAbs is "/Users/me/project/processed"
	// Skip if abs is "/Users/me/project/processed/report.txt"
	if processedAbs != "" && strings.HasPrefix(abs, processedAbs+string(filepath.Separator)) {
		return false
	}

	// The path is a valid file, not hidden, and outside the processed folder. Safe to ingest!
	return true
}
