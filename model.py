import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.block_size, config.embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # Decoder head (tied weights to embeddings)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence too long."

        pos = torch.arange(0, T, device=idx.device)
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.embed_dim, config.num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
        )
        self.ln2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        attn_mask = torch.triu(torch.ones((x.size(1), x.size(1)), device=x.device), 1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_out
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x
