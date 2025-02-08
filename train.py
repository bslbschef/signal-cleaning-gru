import torch
import torch.optim as optim
from utils.data_loader import load_data, prepare_tensor_data
from models.gru_model import GRUNet
from torch.utils.data import DataLoader, TensorDataset
from utils.model_manager import ModelManager

# 超参数设置
window_size = 50  # 假设每个窗口50个时间步
hidden_size = 64
output_size = window_size
num_epochs = 100
batch_size = 32

# 加载训练数据
train_input, train_label = load_data('data/train.mat')
train_data, train_label = prepare_tensor_data(train_input, train_label, window_size)

# 创建数据加载器
train_dataset = TensorDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
model = GRUNet(input_size=2, hidden_size=hidden_size, output_size=output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# 创建ModelManager对象
model_manager = ModelManager(model)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    for data, label in train_loader:
        optimizer.zero_grad()
        output = model(data)  # 模型预测
        loss = criterion(output, label)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        # 每10个epoch保存一次模型
        model_manager.save_model(epoch=epoch + 1)

# 保存最终模型
model_manager.save_model()
