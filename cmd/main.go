package main

import (
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/susanhuhu/go-sentencepiece-encoder/sentencepiece"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Please provide an integer argument.")
		return
	}

	// Get the argument from the command-line
	arg := os.Args[1]

	// Parse the argument as an integer
	count, _ := strconv.Atoi(arg)

	sp, err := sentencepiece.NewSentencepieceFromFile("../sentencepiece/test_data/spm1.model", false)
	if err != nil {
		panic(fmt.Sprintf("Unable to create sentencepiece: %v", err))
	}

	// Create a wait group to wait for all goroutines to finish
	var wg sync.WaitGroup

	// Set the number of goroutines to wait for
	wg.Add(count)

	data := ""
	// Launch count goroutines
	for i := 0; i < count; i++ {
		go func(i int) {
			// Call the method
			start := time.Now()
			sp.TokenizeToOffsets(data)
			latency := time.Since(start)
			fmt.Printf("%d: tokenize %d data used %v \n", i, len(data), latency)

			// Notify the wait group that this goroutine has finished
			wg.Done()
		}(i)
	}

	// Wait for all goroutines to finish
	wg.Wait()
}
