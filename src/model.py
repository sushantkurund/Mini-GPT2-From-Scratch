import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer import Block


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd=384,
        block_size=128,
        n_head=6,
        n_layer=6,
        dropout=0.2
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head,
                    block_size,
                    dropout
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(
            torch.arange(
                T,
                device=idx.device
            )
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)

            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):

        for _ in range(max_new_tokens):

            # Crop context if sequence becomes too long
            idx_cond = idx[:, -self.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Take only the last token prediction
            logits = logits[:, -1, :]

            # Temperature scaling
            logits = logits / temperature

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Append generated token
            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx