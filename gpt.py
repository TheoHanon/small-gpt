from dataclasses import dataclass
from torch import nn
import torch
from attention import MultiHeadAttention

from tokenizers import ByteLevelBPETokenizer


@dataclass
class Config:

    num_heads : int = 4
    d_emb : int = 32
    d_voc : int = 5000
    max_ctx : int = 64
    n_layer : int = 3


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
        self.mlp = MLP(d_emb = config.d_emb)
        self.attn = MultiHeadAttention(config)


    def forward(self, x):

        x = x + self.attn(self.ln_f1(x))
        x = x + self.mlp(self.ln_f2(x))
    
        return x




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


    def forward(self, tokens : torch.Tensor) :

        B, n_ctx = tokens.size()
        assert n_ctx <= self.config.max_ctx, f"Sequence length {n_ctx} is bigger than the maximum context size {self.config.max_ctx}"


        pos = torch.arange(n_ctx, device = tokens.device).unsqueeze(0)
        x = self.enc_tok(tokens) + self.enc_pos(pos)

        for block in self.mha_blocks:
            x = block(x)
        
        logits = self.linear(x)
        return logits




if __name__ == "__main__":



    corpus = [
    "hello world",
    "hi there",
    "how are you",
    "good morning",
    "good night",
    ]


    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(corpus, vocab_size=50, min_frequency=1)

    config = Config(
        d_voc=tokenizer.get_vocab_size()
    )

    model = GPT(config)

    toks =tokenizer.encode("Hello, what is your name ?").ids
    generated = toks

    for _ in range(10):
        x = torch.tensor([generated], dtype=torch.long)  # shape (1, seq_len)
        out = model(x)  # (1, seq_len, d_voc)
        
        # Take logits of the last token only
        last_logits = out[:, -1, :]  # shape (1, d_voc)
        
        # Convert to probabilities
        probs = torch.softmax(last_logits, dim=-1)
        
        # Sample the next token (or use argmax for greedy)
        next_token = torch.argmax(probs, dim=-1).item()
        
        # Append to sequence
        generated.append(next_token)    


    text = tokenizer.decode(generated)
    print(text)
