import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utils.tools import calculate_sqrt_square_sum


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GRUModel, self).__init__()
        # 定义双向 GRU 层，num_layers=2 表示两层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 定义全连接层，用于输出预测值
        # self.fc = nn.Linear(hidden_dim, output_dim)
        # 定义四层 MLP
        self.fc1 = nn.Linear(hidden_dim * 2, 256)  # 由于是双向，hidden_dim 需要乘以 2
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, input_dim)
        # GRU 层的输出
        gru_out, _ = self.gru(x)
        # 选择最后一个时间步的输出进行回归任务
        # output = self.fc(gru_out[:, :, :])  # (batch_size, output_dim)
        # return output
        # 通过四层 MLP
        x = torch.relu(self.fc1(gru_out))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        output = self.fc4(x)  # (batch_size, seq_len, output_dim)
        return output


class WindSpeedCorrectionModel:
    # 注意：序列参数必须在关键词参数前！这是语法！
    def __init__(self, device, train_loader, val_loader, test_loader, input_dim=3, hidden_dim=64, output_dim=3, num_layers=1,
                 lr=0.001, save_interval=20, save_dir='./result/'):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 初始化模型并将模型移到设备上
        self.model = GRUModel(input_dim, hidden_dim, output_dim, num_layers).to(self.device)

        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()  # 回归任务使用均方误差损失
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.save_interval = save_interval
        self.save_dir = save_dir

    @staticmethod
    def convert_to_tensor(data):
        """将numpy数组转换为PyTorch张量"""
        # 先将列表转换为一个numpy数组
        data_array = np.array(data)
        return torch.tensor(data_array, dtype=torch.float32)

    def train_and_validate(self, num_epochs=20):
        self.model.train()  # 设置模型为训练模式

        for epoch in range(num_epochs):
            running_loss = 0.0
            # 训练阶段
            for inputs, labels in self.train_loader:
                # 将数据移到设备上
                # .to(device) 用于将数据（如张量）从 CPU 或 GPU 迁移到指定的设备上，
                # .float() 是用来将数据类型转换为 float32。
                # 神经网络模型（尤其是像 GRU、LSTM 这样的循环神经网络）期望输入数据是 float32 类型。
                # 默认情况下，加载的数据可能会是 int 或其他类型，尤其是当你从文本文件或 CSV 文件加载数据时。这种类型的差异会导致类型不匹配，从而触发错误!
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()

                self.optimizer.zero_grad()  # 清空上一步的梯度
                outputs = self.model(inputs)  # GRU 模型的输出

                # 计算损失
                loss = self.criterion(outputs, labels[:, :, :])  # 使用最后一个时间步的标签进行比较
                running_loss += loss.item()

                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.train_loader)}')

            # 每 20 个 epoch 保存一次模型
            if (epoch + 1) % self.save_interval == 0:
                model_filename = f'model_epoch_{epoch + 1:03d}.pth'
                model_save_path = os.path.join(self.save_dir, model_filename)
                self.save_model(model_save_path)

            # 每个epoch结束后，进行一次验证
            self.validate()
            self.model.train()

    def validate(self):
        self.model.eval()  # 设置模型为评估模式，不会计算梯度
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # 在验证时不计算梯度
            for inputs, labels in self.val_loader:
                # 将数据移到设备上
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # # 假设这是一个分类任务，使用准确率作为评价指标
                # _, predicted = torch.max(outputs, 1)
                # total += labels.size(0)
                # # predicted == labels 的结果是一个布尔类型（bool）的张量，
                # # 而布尔类型的张量本身没有 sum() 方法，因此会导致 Unresolved attribute reference 'sum' for class 'bool' 的警告。
                # correct += (predicted == labels).sum().item()

        val_loss /= len(self.val_loader)
        print(f'Validation Loss: {val_loss}')
        # accuracy = 100 * correct / total
        # print(f'Validation Loss: {val_loss}, Accuracy: {accuracy}%')

    def process_wind_speed_data(self, test_dir, path=None, sequence_length=1200):
        """
        处理test目录下的uav和label数据，分割成指定长度的序列，修正每个序列，
        然后拼接修正后的数据并返回，并生成对比图。

        :param test_dir: 测试数据文件夹路径
        :param sequence_length: 每个序列的长度（默认为1200）
        :return: 修正后的拼接数据
        """
        self.load_model(path)
        # 获取文件夹内的所有uav和label文件
        # x.split('_')：将文件名 x 按照下划线字符 _ 分割成多个部分。
        #
        # x.split('_')[1]：获取分割后的第二部分，即文件名中的数字部分（例如，对于 uav_001.txt，这部分是 '001.txt'）。
        #
        # x.split('_')[1].split('.')[0]：进一步将 '001.txt' 按照点号 . 分割，取第一部分，即 '001'。
        #
        # int(x.split('_')[1].split('.')[0])：将 '001' 转换为整数 1。
        uav_files = [f for f in os.listdir(test_dir) if f.startswith('uav')]
        label_files = [f for f in os.listdir(test_dir) if f.startswith('tower')]
        uav_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        label_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        # 检查文件数量是否一致
        assert len(uav_files) == len(label_files), "uav 和 label 文件数量不匹配"

        # 用于存储修正后的数据
        corrected_data = []

        with torch.no_grad():  # 在测试时不计算梯度
            # 读取每对uav和label文件
            for uav_file, label_file in zip(uav_files, label_files):
                # label_data.append(np.genfromtxt(label_file_path, delimiter=',', skip_header=1))
                uav_data = np.genfromtxt(os.path.join(test_dir, uav_file), delimiter=',', skip_header=1)  # 读取uav数据
                label_data = np.genfromtxt(os.path.join(test_dir, label_file), delimiter=',', skip_header=1)  # 读取label数据

                # 确保uav和label的长度是一样的
                assert len(uav_data) == len(label_data), f"数据长度不匹配: {uav_file}, {label_file}"

                # 将数据按sequence_length分割
                # 列表推导式会遍历 range(0, len(uav_data), sequence_length) 生成的每个整数 i，对于每个 i，
                # 从 uav_data 中截取长度为 sequence_length 的子序列 uav_data[i:i + sequence_length]，
                # 并将这些子序列依次添加到新列表 uav_chunks 中。
                # [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
                uav_chunks = [uav_data[i:i + sequence_length] for i in range(0, len(uav_data), sequence_length)]
                label_chunks = [label_data[i:i + sequence_length] for i in range(0, len(label_data), sequence_length)]

                # # 处理最后一个切片，从后向前划分
                # if len(uav_chunks[-1]) < sequence_length:
                #     # [start:stop:step]; [::-1]当 step 为负数时，表示从后向前进行切片。
                #     # [::-1] 省略了 start 和 stop 参数，步长为 -1，它会将整个序列反转。
                #     uav_chunks[-1] = uav_chunks[-1][::-1]  # 从后向前划分
                #     label_chunks[-1] = label_chunks[-1][::-1]  # 对应的label也需要从后向前划分

                # 处理最后一个切片，从后向前划分
                if len(uav_chunks[-1]) < sequence_length:
                    overlap_size = len(uav_chunks[-1])  # 记录重叠部分的大小

                    # -1 代表最后一个元素，-2 代表倒数第二个元素，依此类推。
                    # [-need_count:] 表示从数组的倒数第 need_count 个元素开始，一直到数组的末尾，把这些元素截取出来。
                    # 例如，如果 need_count 是 100，那么就会从倒数第 100 个元素开始，把这 100 个元素截取出来。
                    uav_borrowed = uav_chunks[-2][-(1200-overlap_size):]  # 从倒数第二个数组借用前k行
                    uav_new_last = np.concatenate([uav_borrowed, uav_chunks[-1]], axis=0)  # 合并
                    uav_chunks[-1] = uav_new_last  # 替换原最后一个数组

                    label_borrowed = label_chunks[-2][-(1200-overlap_size):]  # 从倒数第二个数组借用前k行
                    label_new_last = np.concatenate([label_borrowed, label_chunks[-1]], axis=0)  # 合并
                    label_chunks[-1] = label_new_last  # 替换原最后一个数组

                    # uav_chunks[-1] = uav_chunks[-1][::-1]  # 从后向前划分
                    # label_chunks[-1] = label_chunks[-1][::-1]  # 对应的label也需要从后向前划分
                else:
                    overlap_size = 0  # 如果最后一个块正好是sequence_length大小，则没有重叠

                # 对每个切片进行修正
                for uav_chunk, _ in zip(uav_chunks, label_chunks):
                    uav_chunk_tensor = self.convert_to_tensor(uav_chunk)
                    inputs = uav_chunk_tensor.to(self.device).float()
                    new_inputs = inputs.unsqueeze(0)
                    corrected_chunk = self.model(new_inputs)  # 修正uav数据
                    corrected_data.append(corrected_chunk)

                # 如果有重叠部分，在拼接时移除最后一个块的重复部分
                if overlap_size > 0:
                    corrected_data[-1] = corrected_data[-1][:, (1200-overlap_size):, :]  # 去除重复部分
                    # 使用 unsqueeze 方法在第 0 维插入一个新维度
                    # corrected_data[-1] = corrected_data[-1].unsqueeze(0)

                # 拼接修正后的数据
                # corrected_data是一个list，不能使用np.concatenate()来合并！
                # final_corrected_data = np.concatenate(corrected_data)
                # 在第 0 维上拼接所有张量
                concatenated_tensor = torch.cat(corrected_data, dim=1)
                # 去除多余的维度
                final_corrected_data = concatenated_tensor.view(-1, 3)
                final_corrected_data = torch.tensor(final_corrected_data)
                # 如果张量在 GPU 上，先移动到 CPU
                if final_corrected_data.is_cuda:
                    final_corrected_data = final_corrected_data.cpu()
                # 如果张量需要梯度计算，先调用 detach() 调用 detach() 方法以断开计算图的连接
                if final_corrected_data.requires_grad:
                    final_corrected_data = final_corrected_data.detach()
                # 转换为 NumPy 数组
                # tensor.numpy() 方法仅适用于位于 CPU 上的张量。对于位于 GPU 上的张量，必须先将其移动到 CPU，然后才能调用 numpy() 方法。
                # tensor.numpy() 方法返回的 NumPy 数组与原始张量共享内存，因此对其中一个的修改会影响另一个。
                # 如果不希望共享内存，可以使用 tensor.clone().numpy() 来创建一个新的张量副本。
                final_corrected_data = final_corrected_data.numpy()

                final_uav_chunk = np.concatenate(uav_chunks, axis=0)  #也可以不加axis=0，因为默认是0
                final_label_chunk = np.concatenate(label_chunks, axis=0)

                # 将三个数据数组按列合并成一个二维数组
                # three_data = np.column_stack((final_corrected_data, final_uav_chunk, final_label_chunk))
                # # 定义表头
                # header = "corrected_data uav_data label_data"
                # # 保存修正后的数据，并添加表头
                # np.savetxt(uav_file + '.txt', three_data, header=header)

                # 绘制修正前后与真实标签对比的图像
                plt.figure(figsize=(12, 6))
                uav_uv = calculate_sqrt_square_sum(final_uav_chunk[:,1], final_uav_chunk[:,0])
                cor_uv = calculate_sqrt_square_sum(final_corrected_data[:,1], final_corrected_data[:,0])
                lab_uv = calculate_sqrt_square_sum(final_label_chunk[:,1], final_label_chunk[:,0])
                # plt.plot(final_uav_chunk[:,1], label='修正前（UAV数据）', linestyle='--')
                plt.plot(cor_uv, label='correction')
                plt.plot(lab_uv, label='label', linestyle=':')

                plt.legend()
                plt.xlabel('时间步')
                plt.ylabel('风速值')
                plt.title(f'风速修正对比 - {uav_file}')  # 使用文件名来标识
                break
            plt.show()  # 显示图形

        return final_corrected_data

    def test(self, device, path=None):
        self.load_model(path)

        test_loss = 0.0
        correct = 0
        total = 0
        cnt = 0
        with torch.no_grad():  # 在测试时不计算梯度
            for inputs, labels in self.test_loader:
                cnt += 1
                inputs, labels = inputs.to(device).float(), labels.to(device).float()

                outputs = self.model(inputs)  # 模型预测的输出

                # 计算损失
                loss = self.criterion(outputs, labels)  # 使用最后一个时间步的标签进行比较
                test_loss += loss.item()

                # # 假设这是一个分类任务，使用准确率作为评价指标
                # _, predicted = torch.max(outputs, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

                # 绘制对比图
                if total <= 100:  # 限制显示的数据点数量，不要每个batch都绘制
                    # 提取修正前（uav数据）、修正后（outputs）和真实标签（labels）数据
                    uav_data = inputs[:, -1, :]  # 假设输入是三维的，[batch_size, seq_len, feature_dim]
                    corrected_data = outputs
                    true_labels = labels[:, -1, :]  # 假设使用最后一个时间步的标签

                    # 为每个图生成新的图窗
                    plt.figure(figsize=(12, 6))
                    plt.plot(uav_data.cpu().numpy(), label='修正前（UAV数据）', linestyle='--')
                    plt.plot(corrected_data.cpu().numpy(), label='修正后（模型输出）')
                    plt.plot(true_labels.cpu().numpy(), label='真实标签（Label）', linestyle=':')

                    plt.legend()
                    plt.xlabel('时间步')
                    plt.ylabel('风速值')
                    plt.title(f'风速修正对比 - Epoch {cnt + 1}')

        test_loss /= len(self.test_loader)
        accuracy = 100 * correct / total
        print(f'Test Loss: {test_loss}, Accuracy: {accuracy}%')

        # 最后一次调用 show() 来显示所有图窗
        plt.show()  # 这时会显示所有图窗，避免前后抵消



    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path=None):
        """加载模型"""
        if path:
            # 如果提供了路径，直接加载
            self.model.load_state_dict(torch.load(path))
            print(f'Model loaded from {path}')
        else:
            # 如果没有提供路径，则自动加载序号最大的模型文件
            model_files = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
            # 使用正则表达式提取文件中的序号部分，并按序号降序排列
            # model_files 是一个文件列表，其中包含了所有 .pth 文件的文件名。sort() 方法是用来对这个列表进行排序的。
            # key 参数用于指定一个排序的规则，lambda f: ... 部分定义了排序的方式。
            # reverse=True 表示将排序结果反转，即按照降序排列。如果没有设置 reverse=True，则是升序排序。
            # lambda f: 这里定义了一个匿名函数（lambda 表达式）。f 表示 model_files 列表中的每一个文件名。该匿名函数返回一个整数，作为排序的依据。
            # re.search() 是 Python 正则表达式模块 re 中的一个函数，它用于在字符串 f 中查找匹配正则表达式的内容。
            # r'(\d+)' 是一个正则表达式，表示匹配一串数字：
            # \d 匹配一个数字字符（0-9）。
            # + 表示匹配一个或多个数字字符，确保我们匹配到的数字是连续的。
            # re.search() 会返回一个匹配的结果，如果找到匹配的内容，它会返回一个 match 对象，否则返回 None。
            # match.group() 返回正则表达式匹配到的第一个子串。在这里，group() 会返回文件名中匹配到的数字部分。
            # 例如，如果文件名是 "model001.pth"，re.search(r'(\d+)', f) 会匹配到 "001"，并通过 .group() 返回这个数字。
            # int(...) 将字符串形式的数字转换为整数。比如 "001" 会被转换为整数 1。这样做是为了能够按数值大小对文件进行排序，而不是按字符串字典顺序。
            # reverse=True 表示将文件名按照整数值的降序进行排序。
            # 例如，如果文件名是 "model001.pth", "model003.pth", 和 "model002.pth"，经过排序后，它们会按 003 > 002 > 001 排列。
            model_files.sort(key=lambda f: int(re.search(r'(\d+)', f).group()), reverse=True)
            latest_model = model_files[0]  # 获取最新的模型文件
            latest_model_path = os.path.join(self.save_dir, latest_model)

            self.model.load_state_dict(torch.load(latest_model_path))
            print(f'Model loaded from {latest_model_path}')

        self.model.eval()  # 切换到评估模式
        print(f'Model loaded from {path}')
