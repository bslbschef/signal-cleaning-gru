import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(x)
        if lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)
        return self.fc(out)
