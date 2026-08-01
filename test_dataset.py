from src.dataset import ShakespeareDataset

dataset = ShakespeareDataset(
    "data/tinyshakespeare.txt",
    block_size=128
)

print("Vocabulary:", dataset.vocab_size)

x, y = dataset.get_batch(
    split="train",
    batch_size=64
)

print("Input Shape :", x.shape)
print("Target Shape:", y.shape)