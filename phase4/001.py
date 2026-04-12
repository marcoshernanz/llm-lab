import torch
from torch import nn

VOCAB_SIZE = 128
EMBEDDING_DIM = 64
HIDDEN_DIM = 64


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.hidden = nn.Linear(EMBEDDING_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.hidden(x)
        x = torch.tanh(x)
        x = self.out(x)
        return x
