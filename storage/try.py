# 导入必要的库
from utils.data_loader import WindSpeedCorrectionDataset

# 创建WindSpeedCorrectionDataset类的实例
data_root = './data/half_window_size10'  # 数据根目录
train_dir = '/train'  # 训练数据文件夹
val_dir = '/val'  # 验证数据文件夹
test_dir = '/test'  # 测试数据文件夹
sequence_length = 100  # 序列长度
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

# 测试输出DataLoader的内容
print("Train DataLoader:")
for batch_idx, (uav_data, label_data) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  UAV Data Shape: {uav_data.shape}")
    print(f"  Label Data Shape: {label_data.shape}")
    if batch_idx >= 1:  # 只打印前两个批次的数据
        break

print("\nValidation DataLoader:")
for batch_idx, (uav_data, label_data) in enumerate(val_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  UAV Data Shape: {uav_data.shape}")
    print(f"  Label Data Shape: {label_data.shape}")
    if batch_idx >= 1:  # 只打印前两个批次的数据
        break

print("\nTest DataLoader:")
for batch_idx, (uav_data, label_data) in enumerate(test_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  UAV Data Shape: {uav_data.shape}")
    print(f"  Label Data Shape: {label_data.shape}")
    if batch_idx >= 1:  # 只打印前两个批次的数据
        break
