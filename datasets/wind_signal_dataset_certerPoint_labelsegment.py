import os
import numpy as np
import torch
from torch.utils.data import Dataset

class WindSignalDataset(Dataset):
    def __init__(self, input_dir, label_dir, max_seq_len, test_num=None):
        self.max_seq_len = max_seq_len
        self.inputs = []
        self.labels = []
        self.lengths = []
        # 以下两个变量为局部变量，仅在init函数中出现，init结束后，将不再被需要！
        all_valid_inputs = []
        all_valid_labels = []

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
            input_data = np.loadtxt(input_path, skiprows=1, delimiter=',')
            label_data = np.loadtxt(label_path, skiprows=1, delimiter=',')

            # 处理序列长度
            original_len = input_data.shape[0]

            # 生成以每个数据点为中心的数据段
            for center_idx in range(original_len):
                start_idx = max(0, center_idx - max_seq_len // 2)
                end_idx = min(original_len, center_idx + max_seq_len // 2 + 1)

                # 计算需要填充的长度
                pad_left = max_seq_len // 2 - center_idx
                pad_right = max_seq_len // 2 - (original_len - center_idx - 1)

                # 提取数据段
                segment_input = input_data[start_idx:end_idx]
                segment_label = label_data[start_idx:end_idx]

                # 进行填充
                if pad_left > 0:
                    segment_input = np.pad(segment_input, ((pad_left, 0), (0, 0)), mode='constant', constant_values=0)
                    segment_label = np.pad(segment_label, ((pad_left, 0), (0, 0)), mode='constant', constant_values=0)
                if pad_right > 0:
                    segment_input = np.pad(segment_input, ((0, pad_right), (0, 0)), mode='constant', constant_values=0)
                    segment_label = np.pad(segment_label, ((0, pad_right), (0, 0)), mode='constant', constant_values=0)

                # 收集有效数据用于标准化
                all_valid_inputs.append(segment_input)
                all_valid_labels.append(segment_label)

                # 转换为Tensor
                self.inputs.append(torch.FloatTensor(segment_input))
                self.labels.append(torch.FloatTensor(segment_label))
                self.lengths.append(max_seq_len)
            

        # 计算标准化参数（仅使用有效数据）
        all_valid_inputs = np.concatenate(all_valid_inputs)
        all_valid_labels = np.concatenate(all_valid_labels)

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

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            torch.tensor(self.lengths[idx], dtype=torch.long)
        )
