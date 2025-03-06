import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import math

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, choices=['mlp', 'lstm', 'gru', 'transformer'], default='mlp')
    parser.add_argument('--input_dir', type=str, default='./data_train/inputs')
    parser.add_argument('--label_dir', type=str, default='./data_train/labels')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--max_seq_len', type=int, default=20*60*1)
    return parser.parse_args()

# 数据集类（支持变长序列）
class WindSignalDataset(Dataset):
    def __init__(self, input_dir, label_dir, max_seq_len):
        self.max_seq_len = max_seq_len
        self.inputs = []
        self.labels = []
        self.lengths = []
        # 以下两个变量为局部变量，仅在init函数中出现，init结束后，将不再被需要！
        all_valid_inputs = []
        all_valid_labels = []

        # 第一次遍历：收集有效数据和统计信息
        for fname in sorted(os.listdir(input_dir)):
            input_path = os.path.join(input_dir, fname)
            # label_path = os.path.join(label_dir, fname)
            file_suffix =  fname.split('_')[-1]
            label_fname = f'tower_{file_suffix}'
            label_path = os.path.join(label_dir, label_fname)
            
            # 加载数据
            input_data = np.loadtxt(input_path, skiprows=1, delimiter=',')
            label_data = np.loadtxt(label_path, skiprows=1, delimiter=',')
            
            # 处理序列长度
            original_len = input_data.shape[0]
            effective_len = min(original_len, max_seq_len)
            self.lengths.append(effective_len)

            # 分组和填充
            if original_len > max_seq_len:
                num_segments = (original_len + max_seq_len - 1) // max_seq_len  # 计算需要分段的数量
                for i in range(num_segments):
                    start_idx = i * max_seq_len
                    end_idx = min(start_idx + max_seq_len, original_len)
                    segment_input = input_data[start_idx:end_idx]
                    segment_label = label_data[start_idx:end_idx]

                    # 对最后一段进行填充
                    if end_idx == original_len:
                        pad_len = max_seq_len - (end_idx - start_idx)
                        segment_input = np.pad(segment_input, ((0, pad_len), (0, 0)),
                                               mode='constant', constant_values=0)
                        segment_label = np.pad(segment_label, ((0, pad_len), (0, 0)),
                                               mode='constant', constant_values=0)

                    # 收集有效数据用于标准化
                    all_valid_inputs.append(segment_input)
                    all_valid_labels.append(segment_label)

                    # 转换为Tensor
                    self.inputs.append(torch.FloatTensor(segment_input))
                    self.labels.append(torch.FloatTensor(segment_label))
            else:
                pad_len = max_seq_len - original_len
                padded_input = np.pad(input_data, ((0, pad_len), (0, 0)),
                                      mode='constant', constant_values=0)
                padded_label = np.pad(label_data, ((0, pad_len), (0, 0)),
                                      mode='constant', constant_values=0)
                valid_input = input_data
                valid_label = label_data

                # 收集有效数据用于标准化
                all_valid_inputs.append(valid_input)
                all_valid_labels.append(valid_label)

                # 转换为Tensor
                self.inputs.append(torch.FloatTensor(padded_input))
                self.labels.append(torch.FloatTensor(padded_label))

        # 计算标准化参数（仅使用有效数据）
        all_valid_inputs = np.concatenate(all_valid_inputs)
        all_valid_labels = np.concatenate(all_valid_labels)
        # all_valid_inputs = np.stack(all_valid_inputs)  # 使用 np.stack 保持三维矩阵
        # all_valid_labels = np.stack(all_valid_labels)  # 使用 np.stack 保持三维矩阵
        
        self.input_mean = torch.FloatTensor(np.mean(all_valid_inputs, axis=0))
        self.input_std = torch.FloatTensor(np.std(all_valid_inputs, axis=0))
        self.label_mean = torch.FloatTensor(np.mean(all_valid_labels, axis=0))
        self.label_std = torch.FloatTensor(np.std(all_valid_labels, axis=0))

        # 应用标准化
        for i in range(len(self.inputs)):
            self.inputs[i] = (self.inputs[i] - self.input_mean) / self.input_std
            self.labels[i] = (self.labels[i] - self.label_mean) / self.label_std

    def __len__(self):
        return len(self.inputs)

    # __getitem__ 方法是 Dataset 类中的一个特殊方法，用于从数据集中获取特定索引的数据样本。
    # 在 PyTorch 中，当您使用 DataLoader 来加载数据时，__getitem__ 方法会被自动调用。以下是详细的解释：
    # __getitem__ 方法的作用
    # 获取数据样本: __getitem__ 方法根据给定的索引 idx 返回数据样本。
    # 返回值: 通常返回一个包含输入数据、标签数据和长度信息的元组。
    # __getitem__ 方法的定义
    # 在您的代码中，WindSignalDataset 类继承自 Dataset 类，并重写了 __getitem__ 方法：
    # __getitem__ 方法的调用时机
    # 使用 DataLoader 时:
    # 当您创建 DataLoader 对象并迭代它时，DataLoader 会自动调用 __getitem__ 方法来获取数据样本。
    # DataLoader 会根据 batch_size 和 shuffle 参数来决定如何从数据集中获取样本。
    # 手动调用:
    # 您也可以手动调用 __getitem__ 方法来获取特定索引的数据样本。
    # 例如：datasets[0] 会调用 datasets.__getitem__(0)。
    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            torch.tensor(self.lengths[idx], dtype=torch.long)
        )

# 混合损失函数（支持mask）
class HybridLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        # reduction='none':
        # 含义: 不进行任何聚合操作，返回每个元素的损失值。
        # 输出形状: 与输入张量的形状相同。
        # 用途: 当您需要对每个元素的损失值进行进一步处理（例如，应用掩码或自定义聚合）时，可以使用 reduction='none'。
        # reduction='mean':
        # 含义: 计算所有元素损失值的平均值。
        # 输出形状: 标量（单个浮点数）。
        # 用途: 当您希望得到一个单一的损失值来表示整个批次的损失时，可以使用 reduction='mean'。
        # reduction='sum':
        # 含义: 计算所有元素损失值的总和。
        # 输出形状: 标量（单个浮点数）。
        # 用途: 当您希望得到一个单一的损失值来表示整个批次的损失时，可以使用 reduction='sum'。
        self.mse = nn.MSELoss(reduction='none')

    # 初始化类的时候，会调用init函数；
    # 使用类实例的时候，会自动调用forward函数
    # 在 PyTorch 中，nn.Module 类的子类（如 HybridLoss）中的 forward 方法定义了前向传播的过程。
    # 当您将 HybridLoss 对象作为函数调用时，实际上是在调用其 forward 方法。
    # criterion = HybridLoss(args.alpha)
    # 在训练循环中，criterion(outputs, labels, mask) 实际上调用了 HybridLoss 类的 forward 方法：
    # mask 是一个用于忽略填充部分的掩码张量。
    # 具体来说，mask 用于确保在计算损失时，填充的部分不会对损失值产生影响。
    def forward(self, pred, target, mask):
        # 扩展mask维度
        # mask 是一个形状为 (batch_size, seq_len) 的张量，其中 batch_size 是批量大小，seq_len 是序列长度。
        # mask 中的值为 0 或 1，表示哪些位置是有效的数据，哪些位置是填充的部分。
        # unsqueeze(-1) 在 mask 的最后一个维度上增加一个维度，使其形状变为 (batch_size, seq_len, 1)。
        # 例如，如果 mask 的形状是 (4, 20*60)，经过 unsqueeze(-1) 后，形状变为 (4, 20*60, 1)。
        # expand_as(pred) 将 mask 的形状扩展为与 pred 相同的形状。
        # 假设 pred 的形状是 (batch_size, seq_len, feat_dim)，经过 expand_as(pred) 后，mask 的形状也将变为 (batch_size, seq_len, feat_dim)。
        # 例如，如果 pred 的形状是 (4, 20*60, 3)，经过 expand_as(pred) 后，mask 的形状也将变为 (4, 20*60, 3)。
        # 【维度不对等时，相当于复制3次(4,20*60,1)】
        mask = mask.unsqueeze(-1).expand_as(pred)
        valid_elements = mask.sum()
        
        # 时域损失
        time_loss = (self.mse(pred, target) * mask).sum() / valid_elements
        
        # 频域损失
        pred_fft = torch.fft.fft(pred, dim=1)
        target_fft = torch.fft.fft(target, dim=1)
        freq_loss = (self.mse(torch.abs(pred_fft), torch.abs(target_fft)) * mask).sum() / valid_elements
        
        return self.alpha * time_loss + (1 - self.alpha) * freq_loss

# 模型定义
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, max_seq_len):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(max_seq_len*input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_seq_len*output_dim)
        )
        self.unflatten = nn.Unflatten(1, (max_seq_len, output_dim))
    
    def forward(self, x):
        bs, seq_len, feat_dim = x.shape
        x = self.flatten(x)
        x = self.net(x)
        return self.unflatten(x)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, lengths=None):
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(x)
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.fc(out)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, lengths=None):
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(x)
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.fc(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

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
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(x)

# 训练函数
def train(model, train_loader, val_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = HybridLoss(args.alpha)
    
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels, lengths in train_loader:
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            
            # 生成mask
            batch_size, seq_len = inputs.shape[:2]
            mask = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                mask[i, :lengths[i]] = 1
            
            optimizer.zero_grad()
            
            # 模型前向
            if isinstance(model, Transformer):
                src_key_padding_mask = (mask == 0)
                outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
            elif isinstance(model, (LSTM, GRU)):
                outputs = model(inputs, lengths)
            else:
                outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels, mask)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, lengths in val_loader:
                inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
                
                batch_size, seq_len = inputs.shape[:2]
                mask = torch.zeros(batch_size, seq_len, device=device)
                for i in range(batch_size):
                    mask[i, :lengths[i]] = 1
                
                if isinstance(model, Transformer):
                    src_key_padding_mask = (mask == 0)
                    outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
                elif isinstance(model, (LSTM, GRU)):
                    outputs = model(inputs, lengths)
                else:
                    outputs = model(inputs)
                
                val_loss += criterion(outputs, labels, mask).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1:03d} | Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_{args.model}.pth')

if __name__ == '__main__':
    args = parse_args()
    
    # 数据加载
    dataset = WindSignalDataset(args.input_dir, args.label_dir, args.max_seq_len)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=lambda x: zip(*x))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           collate_fn=lambda x: zip(*x))
    
    # 模型初始化
    input_size = 3
    if args.model == 'mlp':
        model = MLP(input_size, input_size, args.hidden_size, args.max_seq_len)
    elif args.model == 'lstm':
        model = LSTM(input_size, args.hidden_size, args.num_layers)
    elif args.model == 'gru':
        model = GRU(input_size, args.hidden_size, args.num_layers)
    elif args.model == 'transformer':
        model = Transformer(input_size, args.hidden_size, args.nhead,
                           args.num_layers, args.max_seq_len)
    
    # 训练
    train(model, train_loader, val_loader, args)