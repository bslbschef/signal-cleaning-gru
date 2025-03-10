# gru_certerPoint.py
import torch
import torch.nn as nn


class GRUWithMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, mlp_hidden_sizes):
        super(GRUWithMLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU 层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # MLP 层
        mlp_layers = []
        mlp_layers.append(nn.Linear(hidden_size, mlp_hidden_sizes[0]))
        mlp_layers.append(nn.ReLU())
        for i in range(1, len(mlp_hidden_sizes)):
            mlp_layers.append(nn.Linear(mlp_hidden_sizes[i - 1], mlp_hidden_sizes[i]))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(mlp_hidden_sizes[-1], output_size))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 GRU
        output, _ = self.gru(x, h0)
        # 获取最后一个时间步的输出
        last_output = output[:, -1, :]

        # 通过 MLP
        output = self.mlp(last_output)

        # 调整输出形状为 (batch_size, 1, output_size)
        # output = output.unsqueeze(1)
        return output
