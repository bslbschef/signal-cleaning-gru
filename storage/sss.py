def process_wind_speed_data(self, device, test_dir, path=None, sequence_length=1200):
    """
    处理test目录下的uav和label数据，分割成指定长度的序列，修正每个序列，
    然后拼接修正后的数据并返回，并生成对比图。

    :param test_dir: 测试数据文件夹路径
    :param sequence_length: 每个序列的长度（默认为1200）
    :return: 修正后的拼接数据
    """
    self.load_model(path)
    # 获取文件夹内的所有uav和label文件
    uav_files = [f for f in os.listdir(test_dir) if f.startswith('uav')]
    label_files = [f for f in os.listdir(test_dir) if f.startswith('label')]

    # 检查文件数量是否一致
    assert len(uav_files) == len(label_files), "uav 和 label 文件数量不匹配"

    # 用于存储修正后的数据
    corrected_data = []

    with torch.no_grad():  # 在测试时不计算梯度
        # 读取每对uav和label文件
        for uav_file, label_file in zip(uav_files, label_files):
            uav_data = np.loadtxt(os.path.join(test_dir, uav_file))  # 读取uav数据
            label_data = np.loadtxt(os.path.join(test_dir, label_file))  # 读取label数据

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
                uav_chunks[-1] = uav_chunks[-1][::-1]  # 从后向前划分
                label_chunks[-1] = label_chunks[-1][::-1]  # 对应的label也需要从后向前划分
            else:
                overlap_size = 0  # 如果最后一个块正好是sequence_length大小，则没有重叠

            # 对每个切片进行修正
            for uav_chunk, _ in zip(uav_chunks, label_chunks):
                uav_chunk_tensor = self.convert_to_tensor(uav_chunk)
                inputs = uav_chunk_tensor.to(self.device).float()
                corrected_chunk = self.model(inputs)  # 修正uav数据
                corrected_data.append(corrected_chunk)

            # 如果有重叠部分，在拼接时移除最后一个块的重复部分
            if overlap_size > 0:
                corrected_data[-1] = corrected_data[-1][overlap_size:]  # 去除重复部分

            # 拼接修正后的数据
            final_corrected_data = np.concatenate(corrected_data)
            final_uav_chunk = np.concatenate(uav_chunks)
            final_label_chunk = np.concatenate(label_chunks)

            # 将三个数据数组按列合并成一个二维数组
            three_data = np.column_stack((final_corrected_data, final_uav_chunk, final_label_chunk))
            # 定义表头
            header = "corrected_data uav_data label_data"
            # 保存修正后的数据，并添加表头
            np.savetxt(uav_file+'.txt', three_data, header=header)

            # 绘制修正前后与真实标签对比的图像
            plt.figure(figsize=(12, 6))
            plt.plot(final_uav_chunk, label='修正前（UAV数据）', linestyle='--')
            plt.plot(final_corrected_chunk, label='修正后（模型输出）')
            plt.plot(final_label_chunk, label='真实标签（Label）', linestyle=':')

            plt.legend()
            plt.xlabel('时间步')
            plt.ylabel('风速值')
            plt.title(f'风速修正对比 - {uav_file}')  # 使用文件名来标识
            plt.show()  # 显示图形

    return final_corrected_data
