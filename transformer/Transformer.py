"""
The model and hekpful classes. Built bottom-up
"""
import torch
import torch.nn as nn
import math
from torch import tensor

class Attention(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int, ) -> None:
        super().__init__()
        self.Wq = nn.Linear(dim_in, dim_k, dtype=torch.float32)    # learned weights
        self.Wk = nn.Linear(dim_in, dim_k, dtype=torch.float32)
        self.Wv = nn.Linear(dim_in, dim_v, dtype=torch.float32)

    def forward(self, query: tensor, key: tensor, value: tensor) -> tensor:
        query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)
        return scaled_dot_product_attention(self.Wq(query), self.Wk(key), self.Wv(value))
    
def scaled_dot_product_attention(Q: tensor, K: tensor, V: tensor) -> tensor:
    # i think we use batch mm here because for multiple embeddings x 2d weight matric = 3d
    return torch.bmm(torch.nn.functional.softmax(torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5), dim=-1), V.float())

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Attention(dim_in, dim_k, dim_v) for _ in range(num_heads)])
        self.Wo = nn.Linear(num_heads * dim_v, dim_in, dtype=torch.float32)  # learned weights
    
    def forward(self, query: tensor, key: tensor, value: tensor) -> tensor:
        return self.Wo(torch.cat([head(query, key, value) for head in self.heads], dim=-1))    # dim=-1 for concat on last dimension

# encode the position of each token along to give positional information
def positional_encoding(sequence_length: int, d_model: int, device: torch.device = torch.device("cpu")) -> tensor:
    pos = torch.arange(0, sequence_length, dtype=float, device=device).reshape(1, -1, 1)    # col vec
    ind = torch.arange(0, d_model, dtype=float, device=device).reshape(1, 1, -1)    # row vec
    inner_exp = pos / ((10000)**(2*ind / d_model))  # seq x d_model  sized matrix
    return torch.where(ind % 2 == 0, torch.sin(inner_exp), torch.cos(inner_exp))    # sin every even col and cos every odd col

# from paper - two linear layers with relu in between
def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward, dtype=torch.float32),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input, dtype=torch.float32),
    )

# residual sublayer to capture layer norm and dropout
class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, layer_norm_dim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.layernorm = nn.LayerNorm(layer_norm_dim, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, *tensors: tensor) -> tensor:
        # there's a reason for tensors[0]...see decoder block
        x = tensors[0] + self.dropout(self.sublayer(*tensors))
        x = x.to(torch.float32)
        return self.layernorm(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dim_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        dim_k = dim_v = dim_model // num_heads # from paper
        self.attention = Residual(MultiHeadAttention(num_heads, dim_model, dim_k, dim_v), dim_model, dropout)
        self.ff = Residual(feed_forward(dim_model, dim_ff), dim_model, dropout)
    
    def forward(self, x: tensor) -> tensor:
        return self.ff(self.attention(x, x, x))
    
class Encoder(nn.Module):
    def __init__(self, dim_model: int = 512, dim_ff: int = 2048, 
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(dim_model, num_heads, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.device = device

    def forward(self, x: tensor) -> tensor:
        x = x + positional_encoding(x.size(1), x.size(2), device=self.device)   # encode the position
        for layer in self.layers:   # feed forward through each layer
            x = layer(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim_model: int = 512, num_heads: int = 8, dim_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads # from paper
        self.attention1 = Residual(MultiHeadAttention(num_heads, dim_model, dim_k, dim_v), dim_model, dropout)
        self.attention2 = Residual(MultiHeadAttention(num_heads, dim_model, dim_k, dim_v), dim_model, dropout)
        self.ff = Residual(feed_forward(dim_model, dim_ff), dim_model, dropout)

    def forward(self, x: tensor, memory: tensor) -> tensor:
        # x is the embedding from outputs, memory comes from output of Encoder
        x = self.attention1(x, x, x)
        x = self.attention2(x, memory, memory)
        return self.ff(x)

class Decoder(nn.Module):
    def __init__(self, 
                 dim_model: int = 512, dim_ff: int = 2048, 
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(dim_model, num_heads, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.device = device

    def forward(self, x: tensor, memory: tensor) -> tensor:
        x = x + positional_encoding(x.size(1), x.size(2), device=self.device)   # encode the position
        for layer in self.layers:   # feed forward through each layer
            x = layer(x, memory)
        return x
    
class Transformer(nn.Module):
    def __init__(self,
                 num_enc_blocks: int = 6, num_dec_blocks: int = 6, 
                 dim_model: int = 512, dim_ff: int = 2048, 
                 num_heads: int = 8, dropout: float = 0.1,
                 out_distribution_size: int = 1,
                 device: torch.device = torch.device("cpu"),
                 ):
        super().__init__()
        self.encoder = Encoder(dim_model, dim_ff, num_heads, num_enc_blocks, dropout, device=device)
        self.decoder = Decoder(dim_model, dim_ff, num_heads, num_dec_blocks, dropout, device=device)
        self.linear = nn.Linear(dim_model, out_distribution_size, dtype=torch.float32)

    def forward(self, x_in: tensor, x_out: tensor) -> tensor:
        x_in, x_out = x_in.to(torch.int64), x_out.to(torch.int64)
        x = self.decoder(x_out, self.encoder(x_in))
        x = x[:, -1:, :] # take the last token
        return torch.softmax(self.linear(x), dim=-1)