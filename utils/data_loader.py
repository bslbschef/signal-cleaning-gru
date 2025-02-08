import scipy.io as sio
import numpy as np
import torch
from preprocess.fft_processing import preprocess_signal


# 读取.mat文件
def load_data(file_name):
    data = sio.loadmat(file_name)
    input_data = data['input']
    label_data = data['label']
    return input_data, label_data


# 处理数据（将输入信号的实部和虚部分别作为GRU的输入）
def prepare_data(input_data, label_data, window_size):
    real_input = []
    imag_input = []
    label_output = []

    for i in range(len(input_data)):
        input_signal = input_data[i]
        label_signal = label_data[i]

        real_part, imag_part = preprocess_signal(input_signal)

        # 分割数据为窗口
        for start in range(0, len(real_part) - window_size, window_size):
            real_input.append(real_part[start:start + window_size])
            imag_input.append(imag_part[start:start + window_size])
            label_output.append(label_signal[start:start + window_size])

    return np.array(real_input), np.array(imag_input), np.array(label_output)


# 将数据转换为tensor
def prepare_tensor_data(input_data, label_data, window_size):
    real_input, imag_input, label_output = prepare_data(input_data, label_data, window_size)
    real_input = torch.tensor(real_input, dtype=torch.float32)
    imag_input = torch.tensor(imag_input, dtype=torch.float32)
    label_output = torch.tensor(label_output, dtype=torch.float32)
    data = torch.stack([real_input, imag_input], dim=2)  # 合并实部和虚部作为GRU的输入
    return data, label_output
