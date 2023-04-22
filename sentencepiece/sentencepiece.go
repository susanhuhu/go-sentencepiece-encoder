package sentencepiece

import (
	"fmt"
	"math"
	"strings"
	"unicode"
	"unicode/utf8"
)

const minScore float32 = -math.MaxFloat32
const sep rune = 0x2581
const unknown string = "<unk>"

type slice struct {
	score float32
	index int32
	start int
	end   int
}

func findOffset(position int, q string) int {
	count := 0
	for i := range q {
		if count == position {
			return i
		}
	}
	return -1
}

func text(s slice, q string) string {
	startOffset := findOffset(s.start, q)
	endOffset := findOffset(s.end, q)
	if startOffset == -1 || endOffset == -1 {
		return ""
	}
	return q[startOffset:endOffset]
}

type trieNode struct {
	text     string
	level    int
	score    float32
	index    int32
	end      bool
	children map[rune]trieNode
}

func newTrieNode(text string, level int) trieNode {
	return trieNode{
		text:     text,
		level:    level,
		score:    0.0,
		index:    0,
		end:      false,
		children: make(map[rune]trieNode),
	}
}

// Sentencepiece holds the model
type Sentencepiece struct {
	root         trieNode
	lowercase    bool
	unknown      int32
	controlWords map[string]int32
}

// NewEmptySentencepiece creates an empty sentencepiece model
func NewEmptySentencepiece(lowercase bool) Sentencepiece {
	return Sentencepiece{
		root:         newTrieNode("", 0),
		lowercase:    lowercase,
		unknown:      0,
		controlWords: make(map[string]int32),
	}
}

// SetUnknownIndex sets the index for the unknown id
func (s *Sentencepiece) SetUnknownIndex(index int32) {
	s.unknown = index
}

// GetUnknownIndex gets the index of the unknown id
func (s *Sentencepiece) GetUnknownIndex() int32 {
	return s.unknown
}

// SetControlWord sets the index for the given control word
func (s *Sentencepiece) SetControlWord(word string, index int32) {
	s.controlWords[word] = index
}

// GetControlWord gets the index for the given control word
func (s *Sentencepiece) GetControlWord(word string) (int32, bool) {
	v, ok := s.controlWords[word]
	return v, ok
}

// Tokenize tokenizes text into pieces
func (s *Sentencepiece) Tokenize(text string) []Token {
	runes := s.prepareFortokenize(text)
	tokenOffsets := s.tokenizeToOffsets(runes, false)
	return makeTokens(tokenOffsets, runes)
}

// TokenizeToIDs tokenizes text into ids from the vocab
func (s *Sentencepiece) TokenizeToIDs(text string) []int32 {
	tokens := s.Tokenize(text)
	ids := make([]int32, len(tokens))
	for i, token := range tokens {
		ids[i] = token.ID
	}
	return ids
}

func (s *Sentencepiece) TokenizeToOffsets(text string) []TokenOffset {
	runes := s.prepareFortokenize(text)
	padding := len(runes) - len([]rune(text))
	return s.tokenizeToOffsets(runes, padding > 0)
}

func (s *Sentencepiece) tokenizeToOffsets(runes []rune, adjustFirstPadding bool) []TokenOffset {
	slices := s.decodeForwardToken(runes)
	slices = s.decodeBackwards(slices)
	return s.sliceToTokens(slices, runes, adjustFirstPadding)
}

func (s *Sentencepiece) prepareFortokenize(text string) []rune {
	text = normalize(text)
	if s.lowercase {
		text = strings.ToLower(text)
	}
	runes := torunes(text)
	replaceWhiteSpace(runes)
	return runes
}
func (s *Sentencepiece) insert(word string, score float32, index int32) {
	_, size := utf8.DecodeLastRuneInString(word)
	charCount := len(word)
	node := &s.root
	for i, r := range word {
		text := node.text
		cnode, ok := node.children[r]
		if !ok {
			newText := addChar(text, r)
			cnode = newTrieNode(newText, node.level+1)
		}
		if i == charCount-size {
			cnode.end = true
			cnode.score = score
			cnode.index = index
		}
		node.children[r] = cnode
		node = &cnode
	}
}

func (s *Sentencepiece) commonPrefixSearch(runes []rune) []trieNode {
	output := make([]trieNode, 0, len(runes))
	node := &s.root
	for _, r := range runes {
		cnode, ok := node.children[r]
		if !ok {
			break
		}
		if cnode.end {
			output = append(output, cnode)
		}
		node = &cnode
	}
	return output
}

func (s *Sentencepiece) decodeBackwards(slices []slice) []slice {
	best := make([]slice, len(slices))
	len := len(slices) - 1
	i := len
	index := len
	for ; i >= 0; i-- {
		s := slices[index]
		if s.start == -1 {
			i++
			break
		}
		best[i] = s
		index = s.start
	}
	return best[i : len+1]
}

func (s *Sentencepiece) decodeForwardToken(runes []rune) []slice {
	scores := initScores(len(runes) + 1)
	slices := s.initSlices(len(runes) + 1)
	scores[0] = 0.0
	for i := range runes {
		matches := s.commonPrefixSearch(runes[i:])
		for _, node := range matches {
			localScore := scores[i] + node.score
			charEnd := i + node.level
			if localScore > scores[charEnd] {
				slices[charEnd] = slice{score: localScore, index: node.index, start: i, end: charEnd}
				scores[charEnd] = localScore
			}
		}
		if scores[i+1] <= minScore {
			slices[i+1] = slice{score: minScore, index: s.unknown, start: i, end: i + 1}
			scores[i+1] = 0.0
		}
	}
	return slices
}

func (s *Sentencepiece) sliceToTokens(slices []slice, runes []rune, adjustFirstPadding bool) []TokenOffset {
	tokens := make([]TokenOffset, 0, len(slices)+1)
	isPrevUnknown := false
	for _, slice := range slices {
		if !isPrevUnknown || slice.index != s.unknown {
			word := string(runes[slice.start:slice.end])
			start := slice.start
			end := slice.end
			if adjustFirstPadding {
				if start > 0 {
					start -= 1
				}
				end--
			}
			tokens = append(tokens, TokenOffset{ID: slice.index, Text: word, Start: start, End: end})
		}
		isPrevUnknown = slice.index == s.unknown
	}
	return tokens
}

func initScores(len int) []float32 {
	scores := make([]float32, len)
	for i := range scores {
		scores[i] = minScore
	}
	return scores
}

func (s *Sentencepiece) initSlices(len int) []slice {
	slices := make([]slice, len)
	for i := range slices {
		slices[i].start = -1
		slices[i].index = s.unknown
	}
	return slices
}

func replaceWhiteSpace(runes []rune) {
	for i, r := range runes {
		if unicode.IsSpace(r) {
			runes[i] = sep
		}
	}
}

func replaceSeperator(s string) string {
	replacer := func(r rune) rune {
		if r == sep {
			return ' '
		}
		return r
	}
	return strings.Map(replacer, s)
}

func torunes(text string) []rune {
	runes := make([]rune, 0, len(text)+1)

	first, _ := utf8.DecodeRuneInString(text)
	if first != sep {
		runes = append(runes, sep)
	}

	for _, r := range text {
		runes = append(runes, r)
	}

	return runes
}

func makeTokens(offsets []TokenOffset, runes []rune) []Token {
	tokens := make([]Token, len(offsets))
	for i, offset := range offsets {
		tokens[i] = Token{ID: offset.ID, Text: offset.Text}
	}
	return tokens
}

func addChar(s string, r rune) string {
	return fmt.Sprintf("%s%c", s, r)
}
