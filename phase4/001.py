# %%

import torch
from torch import nn

VOCAB_SIZE = 128
EMBEDDING_DIM = 64
HIDDEN_DIM = 64


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_weights = torch.randn(VOCAB_SIZE * EMBEDDING_DIM)
        self.hidden_weights = torch.randn(EMBEDDING_DIM * HIDDEN_DIM)
        self.hidden_biases = torch.randn(HIDDEN_DIM)
        self.out_weights = torch.randn(HIDDEN_DIM * VOCAB_SIZE)
        self.out_biases = torch.randn(VOCAB_SIZE)

    def forward(self, x: torch.Tensor):
        x = x @ self.embedding_weights
        x = x @ self.hidden_weights + self.hidden_biases
        return x @ self.out_weights + self.out_biases
