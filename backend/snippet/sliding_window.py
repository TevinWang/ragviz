
import sys
sys.path.append("/home/tevinw/ragviz/backend")

from snippet.snippet import Snippet
import torch

class SlidingWindowSnippet(Snippet):
    def __init__(self, tokenizer, model, stride, window_size):
        self.tokenizer = tokenizer
        self.model = model
        self.stride = stride
        self.window_size = window_size
    
    def get_snippet(self, query, article):
        input_ids = self.tokenizer(query, return_tensors="pt").input_ids
        decoder_input_ids = self.tokenizer(query, return_tensors="pt").input_ids

        # Forward pass through the model to obtain embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        # Extract the embeddings
        embeddings = outputs.last_hidden_state  # Last layer hidden states

        query_embedding = embeddings[0,0]

        tokens = self.tokenizer.tokenize(article)
        input_ids = self.tokenizer(article, return_tensors="pt").input_ids
        decoder_input_ids = self.tokenizer(article, return_tensors="pt").input_ids

        best_tokens = []
        best_sim = -torch.inf

        for i in range(0, len(input_ids[0]), self.stride):
            cur_input_ids = input_ids[:, i:i+self.window_size]
            cur_decoder_input_ids = decoder_input_ids[:, i:i+self.window_size]

            with torch.no_grad():
                outputs = self.model(input_ids=cur_input_ids, decoder_input_ids=cur_decoder_input_ids)
            
            embeddings = outputs.last_hidden_state

            snippet_embedding = embeddings[0,0]

            sim = float(torch.dot(query_embedding, snippet_embedding))
            if sim > best_sim:
                best_sim = sim
                best_tokens = tokens[i:i+self.window_size]
        
        return self.tokenizer.convert_tokens_to_string(best_tokens)