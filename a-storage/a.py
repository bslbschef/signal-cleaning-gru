import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                      choices=['mlp', 'lstm', 'gru', 'transformer'])
    parser.add_argument('--input_dir', type=str, default='input')
    parser.add_argument('--label_dir', type=str, default='label')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.7,
                      help='Weight for time domain loss')
    return parser.parse_args()

# 数据集类
class WindSignalDataset(Dataset):
    def __init__(self, input_dir, label_dir):
        self.inputs = []
        self.labels = []
        
        # 读取数据
        for fname in os.listdir(input_dir):
            input_path = os.path.join(input_dir, fname)
            label_path = os.path.join(label_dir, fname)
            
            input_data = torch.from_numpy(np.loadtxt(input_path)).float()
            label_data = torch.from_numpy(np.loadtxt(label_path)).float()
            
            self.inputs.append(input_data)
            self.labels.append(label_data)
        
        # 标准化处理
        self.input_mean = torch.stack(self.inputs).mean(dim=(0,1))
        self.input_std = torch.stack(self.inputs).std(dim=(0,1))
        self.label_mean = torch.stack(self.labels).mean(dim=(0,1))
        self.label_std = torch.stack(self.labels).std(dim=(0,1))
        
        for i in range(len(self.inputs)):
            self.inputs[i] = (self.inputs[i] - self.input_mean) / self.input_std
            self.labels[i] = (self.labels[i] - self.label_mean) / self.label_std
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# 混合损失函数（时域+频域）
class HybridLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # 时域损失
        time_loss = self.mse(pred, target)
        
        # 频域损失
        pred_fft = torch.fft.fft(pred, dim=1)
        target_fft = torch.fft.fft(target, dim=1)
        freq_loss = self.mse(torch.abs(pred_fft), torch.abs(target_fft))
        
        return self.alpha*time_loss + (1-self.alpha)*freq_loss

# 模型定义
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim))
    
    def forward(self, x):
        orig_shape = x.shape
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x.view(orig_shape)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, input_size)
    
    def forward(self, x):
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return self.decoder(x)

# 训练函数
def train(model, train_loader, val_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = HybridLoss(args.alpha)
    
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        print(f'Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader):.4f}')

if __name__ == '__main__':
    args = parse_args()
    
    # 数据加载
    dataset = WindSignalDataset(args.input_dir, args.label_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 模型初始化
    sample_input, _ = dataset[0]
    seq_len, input_size = sample_input.shape
    
    if args.model == 'mlp':
        model = MLP(seq_len*input_size, seq_len*input_size, args.hidden_size)
    elif args.model == 'lstm':
        model = LSTM(input_size, args.hidden_size, args.num_layers)
    elif args.model == 'gru':
        model = GRU(input_size, args.hidden_size, args.num_layers)
    elif args.model == 'transformer':
        model = Transformer(input_size, args.hidden_size, args.nhead, args.num_layers)
    
    # 训练
    train(model, train_loader, val_loader, args)

# python main.py --model mlp --input_dir ./input --label_dir ./label --epochs 100