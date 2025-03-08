import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False,
                        choices=['mlp', 'lstm', 'gru', 'transformer'],
                        default='lstm')
    parser.add_argument('--input_dir', type=str, default='./datasets/data_train/inputs')
    parser.add_argument('--label_dir', type=str, default='./datasets/data_train/labels')
    parser.add_argument('--test_input_dir', type=str, default='./datasets/data_test/inputs')
    parser.add_argument('--test_label_dir', type=str, default='./datasets/data_test/labels')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--input_size', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--output_size', type=int, default=4)
    # nargs='+': 表示该参数可以接受一个或多个值，并将这些值作为一个列表返回
    parser.add_argument('--mlp_hidden_sizes', type=int, nargs='+', default=[64, 32, 16], help='MLP hidden sizes')
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--max_seq_len', type=int, default=20*60*1-1)
    parser.add_argument('--k_folds', type=int, default=5)  # 添加 K 折交叉验证的参数
    parser.add_argument('--model_save_dir', type=str, default='./model_checkpoints')  # 添加模型存储地址的参数
    parser.add_argument('--logger_dir', type=str, default='./logger')  # 添加模型存储地址的参数
    parser.add_argument('--result_dir', type=str, default='./result')  # 添加模型存储地址的参数
    return parser.parse_args()