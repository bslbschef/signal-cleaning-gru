import os
import numpy as np
import torch
from torch.utils.data import Dataset

class WindSignalDataset(Dataset):
    def __init__(self, input_dir, label_dir, max_seq_len, test_num=None):
        self.max_seq_len = max_seq_len
        self.inputs = []
        self.labels = []
        # self.lengths = []

        # 获取文件列表并排序
        input_files = sorted(os.listdir(input_dir))
        label_files = sorted(os.listdir(label_dir))

        # 如果 test_num 不是 None，则只处理第 test_num 个文件
        if test_num is not None:
            if test_num < 0 or test_num >= len(input_files):
                raise ValueError(f"test_num {test_num} is out of range. Valid range is 0 to {len(input_files) - 1}.")
            input_files = [input_files[test_num]]
            label_files = [label_files[test_num]]

        # 第一次遍历：收集有效数据和统计信息
        for input_fname, label_fname in zip(input_files, label_files):
            input_path = os.path.join(input_dir, input_fname)
            label_path = os.path.join(label_dir, label_fname)

            # 加载数据
            input_data = np.loadtxt(input_path, skiprows=1, delimiter=',', dtype=np.float32)
            label_data = np.loadtxt(label_path, skiprows=1, delimiter=',', dtype=np.float32)

            # 处理序列长度
            original_len = input_data.shape[0]

            # 修改后的核心逻辑（替换原有收集数据的逻辑）
            sum_input = np.zeros((input_data.shape[1],), dtype=np.float32)  # 按特征维度初始化
            sum_sq_input = np.zeros_like(sum_input)
            sum_label = np.zeros((label_data.shape[1],), dtype=np.float32)  # 假设label是二维的
            sum_sq_label = np.zeros_like(sum_label)
            count = 0

            # 生成以每个数据点为中心的数据段
            for center_idx in range(original_len):
                start_idx = max(0, center_idx - max_seq_len // 2)
                end_idx = min(original_len, center_idx + max_seq_len // 2 + 1)

                # 计算需要填充的长度
                pad_left = max_seq_len // 2 - center_idx
                pad_right = max_seq_len // 2 - (original_len - center_idx - 1)

                # 提取数据段
                segment_input = input_data[start_idx:end_idx]
                segment_label = label_data[center_idx]

                # 进行填充
                if pad_left > 0:
                    segment_input = np.pad(segment_input, ((pad_left, 0), (0, 0)), mode='constant', constant_values=0)
                if pad_right > 0:
                    segment_input = np.pad(segment_input, ((0, pad_right), (0, 0)), mode='constant', constant_values=0)

                # 替换原有收集逻辑为在线统计
                sum_input += np.sum(segment_input, axis=0)  # 按特征维度累加
                sum_sq_input += np.sum(segment_input ** 2, axis=0)  # 按特征维度累加平方
                sum_label += segment_label
                sum_sq_label += segment_label ** 2
                count += segment_input.shape[0]  # 累计样本数

                # 转换为Tensor
                self.inputs.append(torch.FloatTensor(segment_input))
                self.labels.append(torch.FloatTensor(segment_label))
                # self.lengths.append(max_seq_len)

        # 替换原有标准化参数计算
        self.input_mean = torch.FloatTensor(sum_input / count)
        self.input_std = torch.FloatTensor(np.sqrt(sum_sq_input / count - (sum_input / count) ** 2))
        self.label_mean = torch.FloatTensor(sum_label / len(self.labels))
        self.label_std = torch.FloatTensor(np.sqrt(sum_sq_label / len(self.labels) - (sum_label / len(self.labels)) ** 2))

        # 应用标准化
        for i in range(len(self.inputs)):
            self.inputs[i] = (self.inputs[i] - self.input_mean) / self.input_std
            self.labels[i] = (self.labels[i] - self.label_mean) / self.label_std

    def __len__(self):
        return len(self.inputs)


    # 当调用以下for循环时，会自动调用__getitem__方法！！！
    # dataset = WindSignalDataset(input_dir, label_dir, max_seq_len)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    # for batch_inputs, batch_labels, batch_lengths in dataloader:
    #     # 处理每个批次的数据
    #     pass
    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            # torch.tensor(self.lengths[idx], dtype=torch.long)
        )
