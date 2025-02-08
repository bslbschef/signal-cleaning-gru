import torch
from utils.data_loader import load_data, prepare_tensor_data
from models.gru_model import GRUNet
from utils.plot_utils import plot_comparison
from preprocess.fft_processing import inverse_fft
from utils.model_manager import ModelManager
from train import window_size


# 加载测试数据
test_input, test_label = load_data('data/test.mat')
test_data, test_label = prepare_tensor_data(test_input, test_label, window_size)

# 初始化模型
model = GRUNet(input_size=2, hidden_size=64, output_size=window_size)

# 创建ModelManager对象
model_manager = ModelManager(model)

# 加载训练好的模型（选择模型版本）
model_manager.load_model(epoch=100)  # 加载指定epoch的模型，或者不指定加载最终模型
model.eval()

# 进行预测
with torch.no_grad():
    output = model(test_data)

# 将输出信号从频域转换回时域
pred_real = output[:, :, 0].numpy()  # GRU输出的实部
pred_imag = output[:, :, 1].numpy()  # GRU输出的虚部
predicted_signal = inverse_fft(pred_real, pred_imag)

# 可视化修正前后信号对比
for i in range(len(test_input)):
    plot_comparison(test_input[i], predicted_signal[i], case_idx=i+1)
