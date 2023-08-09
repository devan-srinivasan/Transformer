"""
File to adapt transformer for GPT task
"""
import torch.nn as nn
import math
from torch import tensor
from Transformer import Transformer

# for embedding dem vectors yo -- from harvard -- yay I figured out my own way though kinda
class Embedding(nn.Module):
    def __init__(self, d_model: int, d_vocab: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(d_vocab, d_model)
        self.d_model = d_model
        
    def forward(self, x: tensor) -> tensor:
        return self.embedding(x) * math.sqrt(self.d_model)

# this module will basically embed the input for our purposes
class WrappedModel(nn.Module): 
    def __init__(self, transformer: Transformer, d_model: int, d_vocab: int):
        super().__init__()
        self.transformer = transformer
        self.embed_in = Embedding(d_model, d_vocab)
        self.embed_out = Embedding(d_model, d_vocab)

    def forward(self, words_in: tensor, words_out: tensor) -> tensor:
        # embed and form into batch
        input_embedding = self.embed_in(words_in)
        output_embedding = self.embed_out(words_out)
        
        # pass to transformer
        return self.transformer(input_embedding, output_embedding)