import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# A Tiny GPT-like Model
# -------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_heads=2, n_layers=2, block_size=32):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_heads,
                dim_feedforward=4*n_embd,
                activation="gelu"
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.block_size, "Sequence too long!"

        tok_emb = self.token_embedding(idx) # (B, T, n_embd)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x) # (B, T, vocab_size)
        return logits

# -------------------------------
# Training Utilities
# -------------------------------
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # last token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


if __name__ == "__main__":
    # Example: character-level tiny GPT
    text = "hello world"
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return ''.join([itos[i] for i in l])

    vocab_size = len(chars)

    model = TinyGPT(vocab_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = torch.tensor(encode(text), dtype=torch.long).unsqueeze(0)

    for step in range(200):
        logits = model(data[:, :-1])
        loss = F.cross_entropy(logits.view(-1, vocab_size), data[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}, loss: {loss.item():.4f}")

    # Generate text
    context = torch.zeros((1, 1), dtype=torch.long)
    print("Generated:", decode(generate(model, context, max_new_tokens=20)[0].tolist()))
