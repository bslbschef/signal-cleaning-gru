import torch
import argparse
from utils.tools import create_result_folder
from models.gru_model import WindSpeedCorrectionModel  
from utils.data_loader import WindSpeedCorrectionDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Wind Speed Correction Model')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode: train or test')
    parser.add_argument('--data-root', type=str, default='./data/half_window_size10', help='Data root directory')
    parser.add_argument('--train-dir', type=str, default='/train', help='Training data directory')
    parser.add_argument('--val-dir', type=str, default='/val', help='Validation data directory')
    parser.add_argument('--test-dir', type=str, default='/test', help='Test data directory')
    parser.add_argument('--sequence-length', type=int, default=20*60*1, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--input-dim', type=int, default=18, help='Input dimension')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--output-dim', type=int, default=3, help='Output dimension')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of layers in GRU')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-interval', type=int, default=20, help='Save interval')
    parser.add_argument('--num-epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--result-folder', type=str, default='result', help='Result folder name')
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save = create_result_folder(args.result_folder)

    # 实例化 WindSpeedCorrectionDataset
    dataset = WindSpeedCorrectionDataset(
        data_root=args.data_root,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )

    # 获取训练、验证和测试的DataLoader
    train_loader = dataset.create_train_loader()
    val_loader = dataset.create_val_loader()
    test_loader = dataset.create_test_loader()

    # 初始化模型
    model = WindSpeedCorrectionModel(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        save_interval=args.save_interval,
        save_dir=model_save,
    )

    if args.mode == 'train':
        # 训练模型
        model.train_and_validate(num_epochs=args.num_epochs)
    elif args.mode == 'test':
        # 测试模型
        model.process_wind_speed_data(test_dir=args.test_dir)

if __name__ == '__main__':
    main()
