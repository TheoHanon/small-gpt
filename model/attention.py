import torch
import math
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, config) -> None:

        super().__init__()
        assert config.d_emb % config.num_heads == 0, "d_emb must be divisible by num_heads"

        self.d_emb = config.d_emb
        self.num_heads = config.num_heads
        self.d_head = config.d_emb // config.num_heads # => d_values = d_scores = d_emb / num_heads

        self.W_q = nn.Linear(config.d_emb, config.d_emb)
        self.W_k = nn.Linear(config.d_emb, config.d_emb)
        self.W_v = nn.Linear(config.d_emb, config.d_emb)
        self.W_out = nn.Linear(config.d_emb, config.d_emb)

        # self.register_buffer("mask", 
        #                     torch.tril(torch.ones(config.max_ctx, config.max_ctx, dtype = torch.bool)),
        #                     persistent=False)


    def forward(self, X : torch.Tensor) -> torch.Tensor:

        """
        Input:
        -------
            X : (B, N_ctx, d_emb)
        Return:
        -------
            torch.Tensor : (B, N_ctx, d_emb)


        """

        # device = X.device
        B, N_ctx = X.shape[0], X.shape[1]


        Q = self.W_q(X).view(B, N_ctx, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(X).view(B, N_ctx, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(X).view(B, N_ctx, self.num_heads, self.d_head).transpose(1, 2)


        # scores = torch.einsum("bhid, bhjd -> bhij", Q, K) / math.sqrt(self.d_head)
        # scores = scores.masked_fill(self.mask[None, None, :N_ctx, :N_ctx] == 0, float("-inf"))

        # weights = F.softmax(scores, dim = -1) # (B, H, N_ctx, N_ctx)

        # attentions = torch.einsum("bhij, bhjd -> bhid", weights, V)

        attentions = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal= True,
        )
        attentions = attentions.transpose(1, 2).contiguous().view(B, N_ctx, self.d_emb)

        return self.W_out(attentions)








