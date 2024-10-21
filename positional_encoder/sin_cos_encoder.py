import math
import torch
from torch import nn, Tensor

if torch.cuda.is_available():
    # Set default tensor type to CUDA tensors
    torch.set_default_device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

class SinCosTextEncoder(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, init_range):
        super().__init__()
        # TODO: YOUR CODE HERE # (Part 2a)
        # define the encoder
        self.encoder = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
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
        # pass through layer
        return self.encoder(src) * math.sqrt(self.d_model)


class SinCosPosEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()

        # TODO: YOUR CODE HERE # (Part 2a)
        # define ``positional_encoding`` as described in the GoogleDoc instruction and register it
        # shape must match forward: [seq_len, batch_size] so token embedding and pos embedding can be added
        # TODO: ASK: IS BATCH SIZE D_MODEL??
        self.positional_encoding = torch.empty(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(1)  # Add batch dimension

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
        x = x + self.positional_encoding[: x.size(0)]
        return self.dropout(x)
