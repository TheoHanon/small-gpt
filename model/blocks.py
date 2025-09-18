from torch import nn
import torch
import torch.nn.functional as F

from .attention import MultiHeadAttention


class MLP(nn.Module):

    def __init__(self, d_emb, d_hidden=None):

        super().__init__()
        d_hidden = d_hidden or 2 * d_emb

        self.net = nn.Sequential(
            nn.Linear(d_emb, d_hidden), 
            nn.GELU(),
            nn.Linear(d_hidden, d_emb)
        )

    def forward(self, x):
        return self.net(x)
    

class GPTBlock(nn.Module):


    def __init__(self, config):

        super().__init__()

        self.ln_f1 = nn.LayerNorm(config.d_emb)
        self.ln_f2 = nn.LayerNorm(config.d_emb)
        self.mlp = MLP(d_emb = config.d_emb, d_hidden=config.d_hidden)
        self.attn = MultiHeadAttention(config)


    def forward(self, x):

        x = x + self.attn(self.ln_f1(x))
        x = x + self.mlp(self.ln_f2(x))
    
        return x