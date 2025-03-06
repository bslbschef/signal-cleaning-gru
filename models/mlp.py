import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, max_seq_len):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(max_seq_len * input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_seq_len * output_dim)
        )
        self.unflatten = nn.Unflatten(1, (max_seq_len, output_dim))

    def forward(self, x):
        bs, seq_len, feat_dim = x.shape
        x = self.flatten(x)
        x = self.net(x)
        return self.unflatten(x)
