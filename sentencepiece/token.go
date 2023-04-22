package sentencepiece

// Token holds a unit of a tokenized word
type Token struct {
	ID   int32
	Text string
}

type TokenOffset struct {
	ID    int32
	Text  string
	Start int
	End   int
}
