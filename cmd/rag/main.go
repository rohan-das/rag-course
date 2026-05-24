package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"rag-course/app"
	"rag-course/config"
	"syscall"
)

func main() {
	// Cancel everything cleanly on Ctrl-C / SIGTERM. The same context reaches the REPL, so
	// pressing Ctrl-C while the model is replying tears the request
	// down instead of leaving it dangling.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Calling app.Run will actually call chat.RunREPL, which runs the Read-Eval-Print Loop for our llm chat.
	if err := app.Run(ctx, config.Load()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
