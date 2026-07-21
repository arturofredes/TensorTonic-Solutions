from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        start = 0
        n = len(word)
        if n > self.max_word_len:
            return [self.unk_token]
        end = n
        cur_substr = None
        tokens = []
        while start < end:
            if start == 0:
                substr = word[start:end]
            else: 
                substr = "##" + word[start:end]

            if substr in self.vocab.keys():
                cur_substr = substr
                start = end
                end = n
                tokens.append(cur_substr)
            else:
                end = end - 1

        if cur_substr == None:
            return [self.unk_token]
        else:
            return tokens
                
