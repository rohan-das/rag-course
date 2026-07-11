package chat

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"rag-course/llm"
	"rag-course/rag"
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
func RunREPL(ctx context.Context, client *llm.Client, retriever *rag.Retriever, opts Options) error {
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

		// Keep a separate copy of the current conversation that can be modified
		// with additional context before sending it to the LLM.
		//
		// The original history remains unchanged because it represents the
		// actual conversation between the user and assistant.
		turn := history

		// Retrieval is optional. When a retriever exists, it:
		// 1. Rewrites the user question into a better search query if needed.
		// 2. Converts the query into an embedding vector.
		// 3. Searches the vector database.
		// 4. Returns formatted document excerpts.
		if retriever != nil {
			contextText, retErr := retriever.Retrieve(ctx, history)
			if retErr != nil {
				fmt.Fprintln(os.Stderr, "retrieval err: ", retErr)
			} else if contextText != "" {
				// contextText is the formatted retrieval context returned by
				// retriever.Retrieve(). The format of this value is defined in
				// prompt.go by formatContext().
				//
				// prompt.go also includes an example output in the comments
				// showing what the contextText value looks like before it is
				// added to the user message sent to the LLM.
				turn = withInlineContext(history, contextText)
			}
		}

		reply, err := client.ChatStream(ctx, turn, func(s string) {
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

// withInlineContext creates a copy of the conversation history and replaces the
// latest user message in the copy by adding the retrieved document context
// before the original user question.
//
// The original history is intentionally not modified. Retrieval context is only
// needed for the current LLM request. Keeping it out of the permanent history
// avoids sending the same document excerpts again in future conversation turns.
//
// If the latest message is not from the user, there is no user question
// to attach the retrieved context to.
func withInlineContext(history []llm.Message, contextText string) []llm.Message {
	// Nothing to change if there is no conversation or no retrieved context.
	if len(history) == 0 || contextText == "" {
		return history
	}

	// Retrieved context should be added only to the latest user message.
	// If the latest message is not from the user, there is no question to attach it to.
	last := history[len(history)-1]
	if last.Role != "user" {
		return history
	}

	// Create a copy so the original history stays unchanged.
	// This allows us to send a version with retrieval context without storing
	// those excerpts permanently in the conversation.
	out := make([]llm.Message, len(history))
	copy(out, history)

	// Replace the latest user message with:
	// 1. The retrieved document excerpts.
	// 2. A separator.
	// 3. The original user question.
	out[len(out)-1] = llm.Message{
		Role:    "user",
		Content: contextText + "\n\n--- Question ---\n\n" + last.Content,
	}

	return out
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
