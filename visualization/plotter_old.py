import os
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(test_results, model_name, result_dir):
    # 合并所有样本的数据
    all_raw_input = []
    all_modification = []
    all_raw_label = []
    all_lengths = []
    all_losses = []

    for result in test_results:
        all_raw_input.append(result['raw_input'])
        all_modification.append(result['modification'])
        all_raw_label.append(result['raw_label'])
        all_lengths.append(result['length'])
        all_losses.append(result['loss'])

    # 将列表转换为 numpy 数组
    all_raw_input = np.concatenate(all_raw_input, axis=0)
    all_modification = np.concatenate(all_modification, axis=0)
    all_raw_label = np.concatenate(all_raw_label, axis=0)
    all_lengths = np.array(all_lengths)
    total_length = np.sum(all_lengths)

    # 创建图形
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Merged Test Samples - Model: {model_name} - Total Loss: {np.mean(all_losses):.4f}')

    # 绘制每个特征的对比图
    for ch in range(3):
        axs[ch].plot(all_raw_input[:, ch], label='Raw Input', linestyle='--')
        axs[ch].plot(all_modification[:, ch], label='Modification')
        axs[ch].plot(all_raw_label[:, ch], label='Raw Label', linestyle=':')
        axs[ch].set_title(f'Feature {ch + 1}')
        axs[ch].legend()

    # 设置 x 轴范围
    axs[2].set_xlabel('Time Steps')
    axs[0].set_xlim(0, total_length)

    # 保存图形
    plot_filename = os.path.join(result_dir, f'merged_test_samples_model_{model_name}.png')
    plt.savefig(plot_filename)
    plt.close(fig)
