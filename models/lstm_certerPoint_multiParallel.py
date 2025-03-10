# gru_certerPoint.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMWithMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 mlp_hidden_sizes, bidirectional=False, if_stretch=True):
        super(LSTMWithMLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.if_stretch = if_stretch

        # 编码器：由MLP实现，对输入维度
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, 64)
        mlp_input_size = 64

        # 创建尺度拉升矩阵
        self.scale_matrix = torch.tensor([[i + 1 for _ in range(input_size)] for i in range(32)]).float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.scale_matrix = self.scale_matrix.to(device)
        # self.scale_matrix = self.scale_matrix.to(next(self.parameters()).device)
        # self.register_buffer('scale_matrix', self.scale_matrix)

        # 计算GRU输出维度总和
        self.short_out = hidden_size//4
        self.mid_out = hidden_size//2
        self.long_out = hidden_size
        if bidirectional:
            total_gru_out = (self.short_out + self.mid_out + self.long_out) * 2
        else:
            total_gru_out = self.short_out + self.mid_out + self.long_out

        # GRU 层
        self.gru_short = nn.GRU(mlp_input_size, hidden_size=self.short_out, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.gru_mid = nn.GRU(mlp_input_size, hidden_size=self.mid_out, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.gru_long = nn.GRU(mlp_input_size, hidden_size=self.long_out, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        # MLP 层
        mlp_layers = []
        mlp_layers.append(nn.Linear(total_gru_out, mlp_hidden_sizes[0]))
        mlp_layers.append(nn.ReLU())
        for i in range(1, len(mlp_hidden_sizes)):
            mlp_layers.append(nn.Linear(mlp_hidden_sizes[i - 1], mlp_hidden_sizes[i]))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(mlp_hidden_sizes[-1], output_size))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # 处理预处理MLP
        if self.if_stretch:
            weight_scaled = self.linear1.weight * self.scale_matrix
            linear1_out = F.linear(x, weight_scaled, self.linear1.bias)
        else:
            linear1_out = self.linear1(x)
        linear1_out = F.relu(linear1_out)
        linear2_out = F.relu(self.linear2(linear1_out))
        x_processed = linear2_out  # 形状 (batch, seq_len, 64)

        # 初始化三个GRU的隐藏状态
        num_directions = 2 if self.bidirectional else 1
        batch_size = x.size(0)
        h0_short = torch.zeros(num_directions * self.num_layers, batch_size, self.short_out).to(x.device)
        h0_mid = torch.zeros(num_directions * self.num_layers, batch_size, self.mid_out).to(x.device)
        h0_long = torch.zeros(num_directions * self.num_layers, batch_size, self.long_out).to(x.device)

        # 运行三个GRU
        sequence_size = x_processed.shape[1]
        output_short, _ = self.gru_short(x_processed[:,sequence_size-20:sequence_size+20,:], h0_short)
        output_mid, _ = self.gru_mid(x_processed[:,sequence_size-100:sequence_size+100,:], h0_mid)
        output_long, _ = self.gru_long(x_processed, h0_long)

        # 获取最后一个时间步的输出
        last_short = output_short[:, -1, :]
        last_mid = output_mid[:, -1, :]
        last_long = output_long[:, -1, :]

        # 拼接三个GRU的输出
        concat_output = torch.cat([last_short, last_mid, last_long], dim=1)

        # 通过最终MLP
        output = self.mlp(concat_output)

        # 调整输出形状为 (batch_size, 1, output_size)
        # output = output.unsqueeze(1)
        return output
