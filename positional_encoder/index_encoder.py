import math
import torch
from torch import nn, Tensor


class IndexTextEncoder(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, init_range):
        super().__init__()

        # TODO: YOUR CODE HERE # (Part 2c)
        # define the encoder
        self.encoder = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model-1)
        # END OF YOUR CODE #
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.d_model = d_model

    def forward(self, src: Tensor):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.encoder(src) * math.sqrt(self.d_model)


class IndexPosEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()

        # TODO: YOUR CODE HERE # (Part 2c)
        # define ``positional_encoding`` as described in the GoogleDoc instruction and register it
        self.positional_encoding = torch.arange(max_seq_len) / max_seq_len
        # END OF YOUR CODE

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        # TODO: YOUR CODE HERE # (Part 2c)
        # concatenate ``positional_encoding`` to x (be careful of the shape)
        seq_len = x.size(0)
        positional_encoding = self.positional_encoding[:seq_len]
        positional_encoding = positional_encoding.unsqueeze(-1).unsqueeze(-1) # add extra dimensions at the end
        positional_encoding = positional_encoding.expand(x.size(0), x.size(1), 1) # add 1 index to beginning last dim
        x = torch.cat((positional_encoding, x), dim=-1)
        # END OF YOUR CODE #
        return self.dropout(x)
