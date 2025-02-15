import torch
from utils.tools import create_result_folder
from models.gru_model import WindSpeedCorrectionModel
from utils.data_loader import WindSpeedCorrectionDataset

# 创建WindSpeedCorrectionDataset类的实例
data_root = './data/half_window_size10'  # 数据根目录
train_dir = '/train'  # 训练数据文件夹
val_dir = '/val'  # 验证数据文件夹
test_dir = '/test'  # 测试数据文件夹
sequence_length = 1200  # 序列长度
batch_size = 32  # 批次大小

# 实例化 WindSpeedCorrectionDataset
dataset = WindSpeedCorrectionDataset(
    data_root=data_root,
    train_dir=train_dir,
    val_dir=val_dir,
    test_dir=test_dir,
    sequence_length=sequence_length,
    batch_size=batch_size
)

# 获取训练、验证和测试的DataLoader
train_loader = dataset.create_train_loader()
val_loader = dataset.create_val_loader()
test_loader = dataset.create_test_loader()

# 定义设备，如果GPU可用则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save = create_result_folder('result')
# 假设train_loader, val_loader, test_loader已经定义
model = WindSpeedCorrectionModel(device,
                                 train_loader,
                                 val_loader,
                                 test_loader,
                                 input_dim=18,
                                 hidden_dim=64,
                                 output_dim=3,
                                 num_layers=1,
                                 lr=0.001,
                                 save_interval=20,
                                 save_dir=model_save,)

# 训练模型
model.train_and_validate(num_epochs=20)

# 测试模型
model.process_wind_speed_data(test_dir='./data/half_window_size10/test/')



