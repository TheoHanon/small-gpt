from pydantic import BaseModel, Field
from typing import Optional

from torch import nn
import torch
import torch.nn.functional as F

from .blocks import GPTBlock



class GPTConfig(BaseModel):
    num_heads: int = Field(default=4, ge=1, description="Number of attention heads")
    d_emb: int = Field(default=32, ge=1, description="Embedding dimension")
    d_voc: int = Field(default=5000, ge=1, description="Vocabulary size")
    max_ctx: int = Field(default=64, ge=1, description="Maximum context length")
    n_layer: int = Field(default=3, ge=1, description="Number of transformer layers")
    d_hidden : Optional[int] = Field(default=None, description = "Dimension of the hidden mlp layer")
    
    class Config:
        frozen = True 


class GPT(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.config = config
        
        self.enc_tok = nn.Embedding(config.d_voc, config.d_emb)
        self.enc_pos = nn.Embedding(config.max_ctx, config.d_emb)
        
        self.mha_blocks = nn.ModuleList([
            GPTBlock(config) for _ in range(config.n_layer)
        ])

        self.linear = nn.Linear(config.d_emb, config.d_voc, bias = False)
        self.register_buffer("pos", torch.arange(config.max_ctx).unsqueeze(0), persistent=False)
        

    def forward(self, tokens : torch.Tensor) :

        B, n_ctx = tokens.size()
        assert n_ctx <= self.config.max_ctx, f"Sequence length {n_ctx} is bigger than the maximum context size {self.config.max_ctx}"


        pos = self.pos[:, :n_ctx]
        x = self.enc_tok(tokens) + self.enc_pos(pos)

        for block in self.mha_blocks:
            x = block(x)
        
        logits = self.linear(x)
        return logits
    
    @torch.no_grad()
    def generate(self, tokens : torch.Tensor, max_new_tokens : int , temperature : float = 1.0, top_k = None):

        assert temperature <= 1.0 and temperature > 0, f"Temperature {temperature} must be within 0 < T <= 1"


        for _ in range(max_new_tokens):

            tokens_crop = tokens if tokens.size(1) <= self.config.max_ctx else tokens[:, -self.config.max_ctx:]

            logits = self(tokens_crop)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                _, idxs = torch.topk(-logits, max(0, logits.size(-1) - top_k), dim = -1)
                logits.scatter_(dim = -1, index = idxs, value = float("-inf"))

            probs = F.softmax(logits, dim = -1)
            new_token = torch.multinomial(probs, num_samples = 1)
            tokens = torch.cat((tokens, new_token), dim = 1)
            
        return tokens

