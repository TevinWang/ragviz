import sys
sys.path.append("/home/tevinw/ragviz/backend")

from snippet.snippet import Snippet

class NaiveFirstSnippet(Snippet):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_snippet(self, query, article):
        tokens = self.tokenizer.tokenize(article)
        first_128_tokens = tokens[:128]
        first_128_tokens_string = self.tokenizer.convert_tokens_to_string(first_128_tokens)
        return first_128_tokens_string