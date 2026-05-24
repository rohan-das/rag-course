package chat

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"rag-course/llm"
	"strings"
	"sync"
	"time"
)

// Options configures a single REPL session.
type Options struct {
	// SystemPromptFile is the path to a text/markdown file whose
	// contents become the conversation's system message. A missing
	// file is treated as "no system prompt" — not an error.
	SystemPromptFile string
}

// RunREPL drives an interactive chat session on stdin/stdout. Each
// line the user types is appended to a growing slice of llm.Messages
// and sent to the model; the reply is printed and then appended to the
// same history so subsequent turns retain context.
func RunREPL(ctx context.Context, client *llm.Client, opts Options) error {
	// bufio.NewScanner reads full lines until '\n' (unlike fmt.Scanln which breaks on spaces).
	in := bufio.NewScanner(os.Stdin)

	// Safety Net: Expands scanner capacity from default 64KB up to 1MB maximum.
	// This prevents the application from crashing if a user pastes a huge prompt/essay.
	in.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	history, err := seedHistory(opts.SystemPromptFile)
	if err != nil {
		return err
	}

	fmt.Println("Chat session started. Type Q to quit.")

	for {
		fmt.Print("\n> ")

		// in.Scan() blocks and waits for user keyboard input + Enter key.
		if !in.Scan() {
			if err := in.Err(); err != nil {
				return nil
			}
		}

		input := strings.TrimSpace(in.Text())
		if input == "" {
			continue
		}

		if strings.EqualFold(input, "q") || strings.EqualFold(input, "/exit") || strings.EqualFold(input, "exit") || strings.EqualFold(input, "quit") {
			fmt.Println("Goodbye.")
			return nil
		}

		history = append(history, llm.Message{Role: "user", Content: input})

		spin := startSpinner("thinking")

		// stopOnce ensures spin.Stop() is triggered exactly ONE time across the two potential paths below.
		var stopOnce sync.Once

		reply, err := client.ChatStream(ctx, history, func(s string) {
			// CALL 1 (Visual Timing Path): Kills the spinner the exact millisecond the *first*
			// chunk of text arrives from the LLM so the text prints on a clean terminal line.
			stopOnce.Do(spin.Stop)
			fmt.Print(s)
		})

		// CALL 2 (Error Safety Path): If the API fails entirely or network times out, the streaming
		// callback above never fires. This call guarantees the spinner stops instead of spinning forever.
		// If CALL 1 already ran, sync.Once safely treats this as a no-op.
		stopOnce.Do(spin.Stop)
		fmt.Println()

		if err != nil {
			fmt.Fprintln(os.Stderr, "error: ", err)
			history = history[:len(history)-1]
			continue
		}
		history = append(history, reply)
	}
}

type spinner struct {
	stop chan struct{} // Tells the spinner to stop.
	done chan struct{} // Confirms the spinner has finished cleaning up.
	once sync.Once     // Prevents crashes from double-stopping.
}

// startSpinner starts the spinner
func startSpinner(label string) *spinner {
	s := &spinner{stop: make(chan struct{}), done: make(chan struct{})}

	// Spawn background goroutine so the animation runs concurrently while the main thread waits for the LLM.
	go func() {
		// Handshake Step 2: When this goroutine finishes cleaning the screen and exits,
		// close the 'done' channel to unblock the main thread.
		defer close(s.done)

		frames := []string{"|", "/", "-", "\\"}
		t := time.NewTicker(80 * time.Millisecond)
		defer t.Stop()
		i := 0
		for {
			select {
			case <-s.stop:
				// Close-To-Broadcast Trigger: This case fires instantly when 's.stop' is closed.
				// '\r' moves cursor to column 0, '\033[K' wipes the spinner frame off the terminal.
				fmt.Print("\r\033[K")
				return // Exits the goroutine cleanly
			case <-t.C:
				// Carriage return '\r' sends cursor back to start of line to overwrite the previous frame.
				fmt.Printf("\r%s %s", frames[i%len(frames)], label)
				i++
			}
		}
	}()
	return s
}

// Stop stops the spinner.
func (s *spinner) Stop() {
	// s.once protects against 'panic: close of closed channel' if Stop() is invoked multiple times.
	// Closing 's.stop' acts as a broadcast siren notifying the background worker to stop.
	s.once.Do(func() { close(s.stop) })

	// Handshake Step 1: Block and freeze the main thread right here!
	// This forces a context switch to let the background spinner finish its execution and
	// clear the terminal screen BEFORE the main thread tries to print any AI text chunks.
	<-s.done
}

// seedHistory builds the initial conversation slice. When a system
// prompt file is configured and present, its contents become the
// first message; otherwise the slice starts empty.
func seedHistory(path string) ([]llm.Message, error) {
	if path == "" {
		return nil, nil
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read system prompt: %w", err)
	}

	content := strings.TrimSpace(string(data))
	if content == "" {
		return nil, nil
	}

	return []llm.Message{{Role: "system", Content: content}}, nil
}
