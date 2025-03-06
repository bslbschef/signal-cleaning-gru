import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input, (h0, c0))
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 获取最后一个有效时间步的输出
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
        time_dimension = 1  # batch_first=True 时为 1
        idx = idx.unsqueeze(time_dimension)
        output = output.gather(time_dimension, idx).squeeze(time_dimension)

        # 全连接层
        output = self.fc(output)
        return output
