import math
import torch
from torch import nn, Tensor


class SinCosConcatTextEncoder(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, init_range):
        super().__init__()

        # TODO: YOUR CODE HERE # (Part 2b)
        # define the encoder
        self.encoder = nn.Embedding(n_tokens, d_model // 2)
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


class SinCosConcatPosEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()

        # TODO: YOUR CODE HERE # (Part 2b)
        # define ``positional_encoding`` as described in the GoogleDoc instruction and register it
        self.positional_encoding = torch.empty(max_seq_len, d_model // 2)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / d_model))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding[:max_seq_len]

        # Register positional encoding as buffer
        if not hasattr(self, 'positional_encoding'):
            self.register_buffer('positional_encoding', self.positional_encoding)
        # END OF YOUR CODE #

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, embedding_dim]``
        """
        # TODO: YOUR CODE HERE # (Part 2b)
        # # concatenate ``positional_encoding`` to x (be careful of the shape)
        batches = x.size(1)
        seq_len = x.shape[0]
        positional_encoding = self.positional_encoding.unsqueeze(1)
        positional_encoding = positional_encoding.expand(-1, batches, -1) # Expand along batch dimension
        positional_encoding = positional_encoding[:seq_len]
        x = torch.cat((positional_encoding, x), dim=-1)
        # END OF YOUR CODE #
        return self.dropout(x)
