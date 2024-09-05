from tinygrad import Device
from tinygrad import nn, Tensor, TinyJit
import tinygrad
import numpy as np
from tinygrad.dtype import dtypes
from typing import Optional


class Attention:
    def __init__(self, embed_dim: int, n_heads: int):
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads
        self.scale = self.d_head ** 0.5

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        b, n, d = q.shape # assuming q, k, v are same size
        
        q = q.view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.d_head).permute(0, 2, 3, 1)
        v = v.view(b, n, self.n_heads, self.d_head).transpose(1, 2)
        
        attn = q @ k
        attn  = attn / self.scale
    
        attn = attn.softmax(-1)
        out = attn @ v
    
        out = out.transpose(1, 2).view(b, n, d)
        return out


class TransformerEncoderBlock:
    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int|None=None, dropout:float=0.1, activation:str="quick_gelu"):

        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim*3)
        self.self_attn = Attention(self.embed_dim, self.n_heads)

        if ff_dim is None:
            ff_dim = embed_dim*2
        self.ff_dim = ff_dim
        self.ff1 = nn.Linear(self.embed_dim, self.ff_dim)
        self.ff1._activation = getattr(self.ff1, activation)
        self.ff2 = nn.Linear(self.ff_dim, self.embed_dim)

    def __call__(self, x: Tensor, padding_mask: Optional[Tensor]=None) -> Tensor:
        x = self.norm1(x)
        x = self.self_attn(*self.qkv_proj(x).split(self.embed_dim, -1)).dropout(self.dropout) + x
        x = self.ff2(self.ff1(x)._activation().dropout(self.dropout)).dropout(self.dropout) + x
        x = self.norm2(x)

        return x


class TransformerEncoder:
    def __init__(self, num_layers: int, embed_dim: int, n_heads: int, ff_dim: int=None, dropout:float=0.1):
        self.layers = [
            TransformerEncoderBlock(embed_dim, n_heads, ff_dim, dropout) for _ in range(num_layers)
        ]

    def __call__(self, x, padding_mask) -> Tensor:
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x


class TransformerDecoderBlock:
    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int|None=None, dropout:float=0.1, activation:str="quick_gelu"):
        
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_heads = n_heads

        self.tgt_qkv_proj = nn.Linear(self.embed_dim, self.embed_dim*3)
        self.self_attn = Attention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(self.embed_dim)

        self.cross_q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_attn = Attention(self.embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        if ff_dim is None:
            ff_dim = embed_dim*2
        self.ff_dim = ff_dim
        self.ff1 = nn.Linear(self.embed_dim, self.ff_dim)
        self.ff1._activation = getattr(self.ff1, activation)
        self.ff2 = nn.Linear(self.ff_dim, self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)

    def __call__(self, encoded_input: Tensor, target: Tensor) -> Tensor:
        target = self.norm1(target)
        target = self.self_attn(*self.tgt_qkv_proj(target).split(self.embed_dim, -1)).dropout(self.dropout) + target
    
        encoded_input = self.norm2(encoded_input)
        v = self.cross_v_proj(encoded_input)
        k = self.cross_k_proj(encoded_input)
        q = self.cross_q_proj(target)
    
        cross_attn_out = self.cross_attn(q, k, v)
        cross_attn_out = cross_attn_out.dropout(self.dropout)
        target = cross_attn_out + target
    
        target = self.norm3(target)
        ff_out = self.ff1(target)._activation().dropout(self.dropout)
        ff_out = self.ff2(ff_out).dropout(self.dropout)
        target = ff_out + target
    
        return target


class TransformerDecoder:
    def __init__(self, num_layers: int, embed_dim: int, n_heads: int, ff_dim: int|None=None, dropout:float=0.1):
        self.layers = [
            TransformerDecoderBlock(embed_dim, n_heads, ff_dim, dropout) for _ in range(num_layers)
        ]

    def __call__(self, encoded_input, tgt) -> Tensor:
        for layer in self.layers:
            tgt = layer(encoded_input, tgt)
        return tgt



class SinusoidPositionalEncoding:
    def __init__(self, length: int, embed_dim: int):

        self.pos_emb = self.sinusoid_pos_encoding(length, embed_dim)
        self.length = length
        self.embed_dim = embed_dim
        self._batch_size = None

    
    def __call__(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        if self.pos_emb.ndim == 2:
            self.pos_emb = self.pos_emb.expand(b, *self.pos_emb.shape)
        elif self.pos_emb.shape[0] != b:
            self.pos_emb = self.pos_emb[1:].expand(b, *self.pos_emb.shape[1:])
        
        x = x + self.pos_emb[:, :n, :]
        
        return x

    @staticmethod
    def sinusoid_pos_encoding(length: int, embed_dim: int) -> Tensor:
        assert embed_dim % 2 == 0

        pos = Tensor.arange(length, requires_grad=False).unsqueeze(1)
        div = Tensor.arange(0, embed_dim // 2, requires_grad=False)
        div = (div * (-Tensor(10_000, requires_grad=False).log() / embed_dim)).exp()

        sin_emb = (pos * div).sin()
        cos_emb = (pos * div).cos()

        pos_emb = sin_emb.unsqueeze(2).cat(cos_emb.unsqueeze(2), dim=2)
        pos_emb = pos_emb.reshape(length, embed_dim)

        return pos_emb


class Embedding:
    def __init__(self, vocab_size: int, embed_size: int, padding_idx: int = None):
        self.vocab_sz = vocab_size
        self.embed_sz = embed_size
        self.padding_idx = padding_idx
        self.weight = Tensor.glorot_uniform(vocab_size, embed_size)
        
        if self.padding_idx is not None:
            self.weight[self.padding_idx, :] = 0

    def __call__(self, idx: Tensor) -> Tensor:
        embeddings = self.weight[idx]
        
        if self.padding_idx is not None:
            padding_mask = (idx == self.padding_idx).unsqueeze(-1)
            padding_mask = padding_mask.expand(*embeddings.shape)
            embeddings = embeddings.masked_fill(padding_mask, 0.0)
        
        return embeddings


def create_causal_mask(seq_len: int) -> Tensor:
    return Tensor.ones(seq_len, seq_len, requires_grad=False).tril(0).cast(dtypes.bool)