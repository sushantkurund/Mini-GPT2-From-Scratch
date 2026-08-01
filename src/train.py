import os
import time
import torch

from src.dataset import ShakespeareDataset
from src.model import GPTLanguageModel

# =====================================
# Hyperparameters
# =====================================

batch_size = 32
block_size = 128

max_iters = 5000
eval_interval = 50

learning_rate = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# Create folders
# =====================================

os.makedirs("checkpoints", exist_ok=True)

# =====================================
# Load Dataset
# =====================================

dataset = ShakespeareDataset(
    file_path="data/tinyshakespeare.txt",
    block_size=block_size
)

# =====================================
# Create Model
# =====================================

model = GPTLanguageModel(
    vocab_size=dataset.vocab_size,
    block_size=block_size,
    n_embd=128,
    n_head=4,
    n_layer=4,
    dropout=0.2
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print(f"Using device: {device}")

best_loss = float("inf")

# =====================================
# Training
# =====================================

model.train()

start = time.time()

for step in range(max_iters):

    xb, yb = dataset.get_batch("train", batch_size)

    xb = xb.to(device)
    yb = yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    if step % eval_interval == 0:

        model.eval()

        with torch.no_grad():

            xb_val, yb_val = dataset.get_batch(
                "val",
                batch_size
            )

            xb_val = xb_val.to(device)
            yb_val = yb_val.to(device)

            _, val_loss = model(
                xb_val,
                yb_val
            )

        print(
            f"Step {step:4d}/{max_iters} | "
            f"Validation Loss: {val_loss.item():.4f}"
        )

        if val_loss.item() < best_loss:

            best_loss = val_loss.item()

            torch.save(
                model.state_dict(),
                "checkpoints/model.pth"
            )

            print("✓ Best model saved!")

        model.train()

end = time.time()

print("\n===================================")
print("Training completed successfully!")
print("===================================")
print(f"Best Validation Loss : {best_loss:.4f}")
print(f"Training Time        : {(end-start)/60:.2f} minutes")
print("Saved Model          : checkpoints/model.pth")