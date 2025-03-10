import os
import torch
from torch.utils.data import DataLoader
from datasets.wind_signal_dataset_certerPoint import WindSignalDataset
from utils.logger_define import CustomLogger
from utils.args_parser import parse_args
from models.mlp import MLP
from models.lstm_certerPoint import LSTMWithMLP
from models.gru_certerPoint_multiParallel import GRUWithMLP
from models.transformer import Transformer
from testing.tester import test
from visualization.plotter import plot_comparison

if __name__ == '__main__':
    args = parse_args()

    # 确保图片存储目录存在
    os.makedirs(args.result_dir, exist_ok=True)

    # 测试集评估
    test_dataset = WindSignalDataset(args.test_input_dir, args.test_label_dir, args.max_seq_len, test_num=1)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 测试集使用 batch_size=1
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)  # 测试集使用 batch_size=1

    model = None
    if args.model == 'mlp':
        model = MLP(args.input_size, args.output_size, args.hidden_size, args.max_seq_len)
    elif args.model == 'lstm':
        model = LSTMWithMLP(args.input_size, args.hidden_size, args.num_layers, args.output_size,
                            args.mlp_hidden_sizes)
    elif args.model == 'gru':
        model = GRUWithMLP(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.mlp_hidden_sizes)
    elif args.model == 'transformer':
        model = Transformer(args.input_size, args.hidden_size, args.nhead,
                            args.num_layers, args.max_seq_len)

    # 加载最佳模型
    best_model_path = os.path.join(args.model_save_dir, f'best_{args.model}_fold_1.pth')  # 假设使用第一个折的最佳模型
    model.load_state_dict(torch.load(best_model_path))
    # model.load_state_dict(torch.load(best_model_path), weights_only=True)
    model.eval()

    # 测试
    test_results = test(model, test_loader, args)

    # 绘制对比图
    plot_comparison(test_results, args.model, args.result_dir)
