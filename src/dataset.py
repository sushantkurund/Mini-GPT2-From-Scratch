import torch
from src.tokenizer import CharacterTokenizer


class ShakespeareDataset:
    """
    Loads Tiny Shakespeare dataset and prepares training batches.
    """

    def __init__(self, file_path, block_size=128):

        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.tokenizer = CharacterTokenizer(self.text)

        self.vocab_size = self.tokenizer.vocab_size

        self.data = torch.tensor(
            self.tokenizer.encode(self.text),
            dtype=torch.long
        )

        n = int(0.9 * len(self.data))

        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        self.block_size = block_size

    def get_batch(self, split, batch_size):

        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(
            len(data) - self.block_size,
            (batch_size,)
        )

        x = torch.stack(
            [data[i:i+self.block_size] for i in ix]
        )

        y = torch.stack(
            [data[i+1:i+self.block_size+1] for i in ix]
        )

        return x, y