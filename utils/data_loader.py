import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class WindSpeedCorrectionDataset:
    def __init__(self,
                 data_root='./data/half_window_size10',
                 train_dir='/train',
                 val_dir='/val',
                 test_dir='/test',
                 sequence_length=100,
                 batch_size=32):
        self.train_dir = data_root + train_dir
        self.val_dir = data_root + val_dir
        self.test_dir = data_root + test_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size

    @staticmethod
    def load_data(data_dir):
        """加载指定目录的数据"""
        # os.listdir()返回该路径下所有文件和文件夹的名称列表
        # sorted()用于对可迭代对象（如列表）进行排序，返回一个新的已排序的列表，原可迭代对象不会被修改。
        # startswith() 是 Python 字符串对象的一个内置方法，用于检查字符串是否以指定的前缀开头
        label_files = sorted([f for f in os.listdir(data_dir) if f.startswith('tower_')])
        uav_files = sorted([f for f in os.listdir(data_dir) if f.startswith('uav_')])

        label_data = []
        uav_data = []

        # 读取所有的label数据和uav数据
        # zip() 函数返回一个迭代器，该迭代器生成元组
        for label_file, uav_file in zip(label_files, uav_files):
            label_file_path = os.path.join(data_dir, label_file)
            uav_file_path = os.path.join(data_dir, uav_file)

            # 假设数据是以换行分隔的文本，且每行包含三个值（u, v, w风速）
            # np.loadtxt() 是 numpy 库中的一个函数，用于从文本文件中加载数据
            # skiprows=1表示跳过第一行
            # label_data.append(np.loadtxt(label_file_path, skiprows=1))
            # uav_data.append(np.loadtxt(uav_file_path, skiprows=1))
            label_data.append(np.genfromtxt(label_file_path, delimiter=',', skip_header=1))
            uav_data.append(np.genfromtxt(uav_file_path, delimiter=',', skip_header=1))

        # 将数据堆叠成numpy数组
        # 转换为 NumPy 数组的对象。它可以是列表、元组、其他数组、生成器等可迭代对象。
        # label_data = np.array(label_data)
        # uav_data = np.array(uav_data)

        return label_data, uav_data

    @staticmethod
    def convert_to_tensor(data):
        """将numpy数组转换为PyTorch张量"""
        # 先将列表转换为一个numpy数组
        data_array = np.array(data)
        return torch.tensor(data_array, dtype=torch.float32)

    def split_into_sequences(self, data, type):
        """将数据分为多个序列，每个序列的长度为sequence_length"""
        sequences = []
        for file_data in data:
            # 计算序列数量：//是去尾法！
            num_sequences = len(file_data) // self.sequence_length
            remainder = len(file_data) % self.sequence_length

            # 正常的分段
            for i in range(num_sequences):
                # 获取第 1, 3, 和 10 维度的数据，索引分别是 0, 2, 和 9
                # selected_data = file_data[:, [0, 2, 9]]  # 选择第 1, 3 和 10 列
                if type != 'uav':
                    sequences.append(file_data[i * self.sequence_length: (i + 1) * self.sequence_length, [0, 1, 3]])
                else:
                    sequences.append(file_data[i * self.sequence_length: (i + 1) * self.sequence_length])

            # 处理最后一段，如果不足sequence_length，从后向前截取
            if remainder > 0:
                # is not：用于判断两个对象是否是不同的对象（即它们在内存中的地址是否不同），这通常用于比较对象的身份或是否是 None。
                # !=：用于判断两个值是否不同，这是比较字符串、数字等数据类型的正确方法。
                if type != 'uav':
                    sequences.append(file_data[-self.sequence_length:, [0, 1, 3]])
                else:
                    sequences.append(file_data[-self.sequence_length:])

        # 将数据整理为numpy数组
        return np.array(sequences)

    def prepare_data(self, data_dir):
        """准备数据，将数据加载并分成序列"""
        label_data, uav_data = self.load_data(data_dir)

        label_sequences = []
        uav_sequences = []

        # 对每个文件的数据分别进行切分
        # append()方法：添加到列表末尾的元素，可以是任意数据类型，如整数、字符串、列表、元组等
        # list 对象的 extend() 方法用于在列表的末尾一次性追加另一个可迭代对象（如列表、元组、字符串等）中的所有元素，从而扩展原列表的内容。
        # append() 会将参数作为一个整体添加到列表末尾。
        # extend() 会将一个可迭代对象中的每个元素逐个添加到列表末尾。
        for i in range(len(label_data)):
            # [label_data[i]] 是将 label_data[i] 包装成一个包含单一元素的列表，确保在split_into_sequences函数里不会报错！
            label_sequences.extend(self.split_into_sequences([label_data[i]], type='label'))
            uav_sequences.extend(self.split_into_sequences([uav_data[i]], type='uav'))

        # 转换为PyTorch张量
        label_tensor = self.convert_to_tensor(label_sequences)
        uav_tensor = self.convert_to_tensor(uav_sequences)

        return uav_tensor, label_tensor

    def create_data_loader(self, data_dir, type):
        """根据给定的文件夹路径创建DataLoader"""
        uav_tensor, label_tensor = self.prepare_data(data_dir)

        # 使用TensorDataset包装数据
        dataset = TensorDataset(uav_tensor, label_tensor)

        if type != 'test':
            # 创建DataLoader，支持批量加载和打乱数据
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            # 创建DataLoader，支持批量加载和打乱数据
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        return data_loader

    def create_train_loader(self):
        """创建训练数据加载器"""
        return self.create_data_loader(self.train_dir, 'train')

    def create_val_loader(self):
        """创建验证数据加载器"""
        return self.create_data_loader(self.val_dir, 'val')

    def create_test_loader(self):
        """创建测试数据加载器"""
        return self.create_data_loader(self.test_dir, 'test')
