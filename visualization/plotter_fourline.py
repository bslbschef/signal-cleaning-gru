import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(test_results, model_name, result_dir):
    # 合并所有样本的数据
    all_raw_input = []
    all_modification = []
    all_raw_label = []
    all_loss = []

    for result in test_results:
        all_raw_input.append(result['raw_input'])
        all_modification.append(result['modification'])
        all_raw_label.append(result['raw_label'])
        all_loss.append(result['loss'])

    # 将列表转换为 numpy 数组
    all_raw_input = np.array(all_raw_input)
    all_modification = np.array(all_modification)
    all_raw_label = np.array(all_raw_label)
    all_loss = np.array(all_loss)
    total_length = len(all_raw_input)

    # 绘制每个特征的对比图
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    feature_names = ['u', 'v', 'uv', 'w']
    for ch in range(4):
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f'Merged Test Samples - Model: {model_name} - Feature {feature_names[ch]} - Total Loss: {np.mean(all_loss):.4f}')

        ax.plot(all_raw_input[:, ch], label='Raw Input', linestyle='--')
        ax.plot(all_modification[:, ch], label='Modification')
        ax.plot(all_raw_label[:, ch], label='Raw Label', linestyle=':')
        ax.set_title(f'Feature {ch + 1}')
        ax.legend()

        # 设置 x 轴范围
        ax.set_xlabel('Time Steps')
        ax.set_xlim(0, total_length)

        # 保存图形
        plot_filename = os.path.join(result_dir, f'test_model_{model_name}_feature_{feature_names[ch]}_{timestamp}.png')
        plt.savefig(plot_filename)
        plt.close(fig)
