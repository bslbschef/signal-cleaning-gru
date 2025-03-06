import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True  # 设置为双向LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if self.bidirectional else hidden_size, output_size)  # 双向LSTM需要乘以2

    def forward(self, x, lengths):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input, (h0, c0))
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # 全连接层处理每个时间步的输出
        output = self.fc(output)

        return output
