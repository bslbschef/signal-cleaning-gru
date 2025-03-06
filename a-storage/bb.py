import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 数据集类：加载txt文件并进行FFT处理（沿时间轴，即第0维），并控制序列长度
class WindSpeedDataset(Dataset):
    def __init__(self, sample_dir, label_dir, seq_len, transform_fft=True):
        self.sample_files = sorted([os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith('.txt')])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.txt')])
        assert len(self.sample_files) == len(self.label_files), "样本与标签数量不一致"
        self.seq_len = seq_len
        self.transform_fft = transform_fft

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # 加载数据，假设每个txt文件存储的是 n*3 数组
        sample = np.loadtxt(self.sample_files[idx])
        label = np.loadtxt(self.label_files[idx])
        # 转换为Tensor
        sample = torch.tensor(sample, dtype=torch.float32)  # shape: (n, 3)
        label = torch.tensor(label, dtype=torch.float32)

        # 控制序列长度
        n = sample.shape[0]
        if n >= self.seq_len:
            # 这里直接取前 seq_len 个时间步，如果需要可改为随机截取
            sample = sample[:self.seq_len, :]
            label = label[:self.seq_len, :]
        else:
            # 长度不足时进行零填充（在时间维度上填充到 seq_len）
            pad_len = self.seq_len - n
            sample = F.pad(sample, (0, 0, 0, pad_len))  # (left, right, top, bottom)
            label = F.pad(label, (0, 0, 0, pad_len))

        if self.transform_fft:
            # 对每个通道做FFT（沿着时间维度，即第0维）
            sample_fft = torch.fft.fft(sample, dim=0)  # 得到 shape: (seq_len, 3) 的复数张量
            label_fft  = torch.fft.fft(label, dim=0)
            # 将复数张量分解成实部和虚部，并在特征维度上拼接，变为 (seq_len, 6)
            sample_fft = torch.cat((sample_fft.real, sample_fft.imag), dim=1)
            label_fft  = torch.cat((label_fft.real, label_fft.imag), dim=1)
            return sample_fft, label_fft

        return sample, label

# 定义基类（可用于扩展更多共同操作）
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

# 1. MLP模型：将(seq_len,6)数据展平后进行全连接处理，再reshape回原形
class MLPModel(BaseModel):
    def __init__(self, seq_len, hidden_dim):
        super(MLPModel, self).__init__()
        input_dim = seq_len * 6  # 每个样本展平后总特征数
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.seq_len = seq_len

    def forward(self, x):
        # x shape: (batch_size, seq_len, 6)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        out = self.net(x)
        out = out.view(batch_size, self.seq_len, 6)
        return out

# 2. LSTM模型
class LSTMModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# 3. GRU模型
class GRUModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out

# 4. Transformer模型
class TransformerModel(BaseModel):
    def __init__(self, input_dim, nhead, num_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        # TransformerEncoderLayer 的 d_model 即为输入特征数（这里为6）
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # Transformer需要输入维度为 (seq_len, batch_size, input_dim)
        x = x.transpose(0, 1)
        out = self.transformer_encoder(x)
        out = out.transpose(0, 1)
        out = self.fc(out)
        return out

# Trainer 封装训练循环
class Trainer:
    def __init__(self, model, train_loader, lr, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for data, target in self.train_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(self.train_loader):.4f}")

# 主函数，使用 argparse 解析参数，并根据所选模型构建网络
def main():
    parser = argparse.ArgumentParser(description='风速信号频域去噪任务')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'lstm', 'gru', 'transformer'], help='选择模型类型')
    parser.add_argument('--sample_dir', type=str, default='samples', help='存放样本txt文件的目录')
    parser.add_argument('--label_dir', type=str, default='labels', help='存放标签txt文件的目录')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    # 模型相关参数
    parser.add_argument('--seq_len', type=int, default=100, help='控制每个样本的序列长度')
    parser.add_argument('--hidden_dim', type=int, default=64, help='MLP, LSTM, GRU 的隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM和GRU的层数')
    parser.add_argument('--nhead', type=int, default=2, help='Transformer中的注意力头数')
    parser.add_argument('--dim_feedforward', type=int, default=128, help='Transformer中前馈网络的维度')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集与数据加载器，传入控制序列长度的参数
    dataset = WindSpeedDataset(args.sample_dir, args.label_dir, seq_len=args.seq_len, transform_fft=True)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = 6  # FFT后，每个时刻有6个特征（3通道的实部+虚部）

    # 根据模型类型构造不同模型
    if args.model == 'mlp':
        model = MLPModel(seq_len=args.seq_len, hidden_dim=args.hidden_dim)
    elif args.model == 'lstm':
        model = LSTMModel(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.model == 'gru':
        model = GRUModel(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.model == 'transformer':
        model = TransformerModel(input_dim=input_dim, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward)
    
    trainer = Trainer(model, train_loader, lr=args.lr, device=device)
    trainer.train(args.epochs)

if __name__ == '__main__':
    main()
