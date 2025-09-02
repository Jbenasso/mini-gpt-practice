import torch
import torch.optim as optim
from config import Config
from model import GPT
import os

def train(dataset):
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_iters)

    for step in range(config.max_iters):
        xb, yb = dataset.get_batch(config.batch_size, config.block_size)
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % config.eval_interval == 0:
            print(f"Step {step}, loss {loss.item():.4f}")
            save_checkpoint(model, optimizer, step, config)

    return model


def save_checkpoint(model, optimizer, step, config):
    os.makedirs(config.ckpt_dir, exist_ok=True)
    path = os.path.join(config.ckpt_dir, f"model_step{step}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step
    }, path)
    print(f"Saved checkpoint to {path}")
