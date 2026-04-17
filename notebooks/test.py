# %%

import torch
import math

EMBEDDING_DIM = 4
SEQUENCE_LEN = 8


positions = torch.arange(SEQUENCE_LEN)  # [SEQUENCE_LEN]
pair_ids = torch.arange(0, EMBEDDING_DIM, 2)  # [EMBEDDING_DIM/2]

frequencies = 1.0 / 10000.0 ** (pair_ids / EMBEDDING_DIM) # [EMBEDDING_DIM/2]
angles = frequencies[:, None] * positions[None, :]  # [EMBEDDING_DIM/2, SEQUENCE_LEN]

position_embeddings = torch.zeros(EMBEDDING_DIM, SEQUENCE_LEN)
position_embeddings[0::2, :] = torch.sin(angles)
position_embeddings[1::2, :] = torch.cos(angles)


