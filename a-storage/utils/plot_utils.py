import matplotlib.pyplot as plt

# 绘制修正前后的信号对比
def plot_comparison(original_signal, predicted_signal, case_idx):
    plt.figure()
    plt.plot(original_signal, label='Original Signal')
    plt.plot(predicted_signal, label='Denoised Signal')
    plt.legend()
    plt.title(f"Case {case_idx} Comparison")
    plt.show()
