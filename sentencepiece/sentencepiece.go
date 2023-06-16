package sentencepiece

import (
	"fmt"
	"math"
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

type trieNode struct {
	text     string
	level    int
	score    float32
	index    int32
	end      bool
	children map[rune]trieNode
}

type trieNodeMeta struct {
	level int
	score float32
	index int32
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
	runes := make([]rune, 0, len(text)+1)
	first, _ := utf8.DecodeRuneInString(text)
	if first != sep {
		runes = append(runes, sep)
	}

	for _, r := range text {
		if isControl(r) || r == 0 {
			runes = append(runes, ' ')
		} else if unicode.IsSpace(r) {
			runes = append(runes, sep)
		} else if s.lowercase {
			runes = append(runes, unicode.ToLower(r))
		} else {
			runes = append(runes, r)
		}
	}

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

func (s *Sentencepiece) commonPrefixSearch(runes []rune) []trieNodeMeta {
	var output []trieNodeMeta
	node := s.root
	for _, r := range runes {
		cnode, ok := node.children[r]
		if !ok {
			break
		}
		if cnode.end {
			output = append(output, trieNodeMeta{cnode.level, cnode.score, cnode.index})
		}
		node = cnode
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
			if end > 0 {
				tokens = append(tokens, TokenOffset{ID: slice.index, Text: word, Start: start, End: end})
			}
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

func isControl(c rune) bool {
	if c == ' ' || c == '\n' || c == '\r' || c == '\t' {
		return false
	}
	if c <= 0x001F || (c >= 0x0080 && c <= 0x009F) ||
		(c >= 0xE0020 && c <= 0xE007F) ||
		(c >= 0xE000 && c <= 0xF8FF) ||
		(c >= 0xF0000 && c <= 0xFFFFD) ||
		(c >= 0x100000 && c <= 0x10FFFD) ||
		(c >= 0xD800 && c <= 0xDB7F) ||
		(c >= 0xDB80 && c <= 0xDBFF) ||
		(c >= 0xDC00 && c <= 0xDFFF) ||
		isControlChar(c) {
		return true
	}
	return false
}

func isControlChar(c rune) bool {
	controlChars := []rune{
		0x007F, 0x00AD, 0x0600, 0x0601, 0x0602, 0x0603, 0x0604, 0x0605, 0x061C, 0x06DD, 0x070F,
		0x08E2, 0x180E, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D,
		0x202E, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0x2066, 0x2067, 0x2068, 0x2069, 0x206A,
		0x206B, 0x206C, 0x206D, 0x206E, 0x206F, 0xFEFF, 0xFFF9, 0xFFFA, 0xFFFB, 0x110BD,
		0x110CD, 0x13430, 0x13431, 0x13432, 0x13433, 0x13434, 0x13435, 0x13436, 0x13437,
		0x13438, 0x1BCA0, 0x1BCA1, 0x1BCA2, 0x1BCA3, 0x1D173, 0x1D174, 0x1D175, 0x1D176,
		0x1D177, 0x1D178, 0x1D179, 0x1D17A, 0xE0001,
	}
	for _, ch := range controlChars {
		if ch == c {
			return true
		}
	}
	return false
}
