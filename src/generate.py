import torch

from src.dataset import ShakespeareDataset
from src.model import GPTLanguageModel

# -----------------------
# Configuration
# -----------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 128

# Model architecture (must match training)
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

# -----------------------
# Load Dataset & Tokenizer
# -----------------------

dataset = ShakespeareDataset(
    "data/tinyshakespeare.txt",
    block_size=block_size
)

# -----------------------
# Load Model
# -----------------------

model = GPTLanguageModel(
    vocab_size=dataset.vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/model.pth",
        map_location=device
    )
)

model.eval()

print("✓ Model loaded successfully!")

# -----------------------
# Prompt
# -----------------------

prompt = input("\nEnter a prompt (leave blank for random): ")

if prompt.strip() == "":
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
else:
    encoded = dataset.tokenizer.encode(prompt)
    context = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

# -----------------------
# Generate
# -----------------------

with torch.no_grad():

    generated = model.generate(
        context,
        max_new_tokens=600,
        temperature=0.8
    )

text = dataset.tokenizer.decode(
    generated[0].tolist()
)

print("\n" + "=" * 60)
print("Generated Text")
print("=" * 60)
print(text)