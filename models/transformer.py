import torch
import torch.nn as nn
from models.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, input_size)

    def forward(self, x, src_key_padding_mask=None):
        x = self.embed(x) * torch.sqrt(torch.tensor(self.d_model))
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(x)
