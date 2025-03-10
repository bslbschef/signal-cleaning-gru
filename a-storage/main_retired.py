import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from datasets.wind_signal_dataset_certerPoint import WindSignalDataset
from training.trainer import train
from utils.logger_define import CustomLogger
from utils.args_parser import parse_args
from models.mlp import MLP
from models.lstm_certerPoint import LSTMWithMLP
from models.gru_certerPoint_multiParallel import GRUWithMLP
from models.transformer import Transformer



if __name__ == '__main__':
    args = parse_args()

    # make sure logger save directory exists!
    os.makedirs(args.logger_dir, exist_ok=True)
    # create logger!
    logger = CustomLogger(args.model, args.logger_dir).get_logger()

    # make sure model save directory exists!
    os.makedirs(args.model_save_dir, exist_ok=True)
    logger.info(f'Model save directory: {args.model_save_dir}')

    # dataset initialization!
    dataset = WindSignalDataset(args.input_dir, args.label_dir, args.max_seq_len)
    logger.info(f'Total number of samples: {len(dataset)}')

    # # K 折交叉验证【k=1时，表示整个数据集既是训练集又是测试集!】
    # k_folds = args.k_folds
    # # random_state 是 KFold 类中的一个参数，用于控制数据集划分的随机性。
    # # 具体来说，random_state 用于设置随机种子，确保每次运行代码时，数据集的划分方式是相同的。这对于可重复性和调试非常有用！
    # # 默认值：None，表示不设置随机种子，每次运行时划分顺序可能不同。
    # kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # judge if use k-fold cross validation!
    if args.if_kfold:
        k_folds = args.k_folds
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        splits = kfold.split(dataset)
    else:
        total_indices = list(range(len(dataset)))
        splits = [(total_indices, total_indices)]  # 全量数据作为 train 和 val
        k_folds = 1  # 确保日志显示正确

    # 用于存储每个折的验证损失
    results = {}

    for fold, (train_ids, val_ids) in enumerate(splits):
        logger.info(f'Fold {fold + 1}/{k_folds}')
        logger.info(f'Number of train samples: {len(train_ids)}')
        logger.info(f'Number of val samples: {len(val_ids)}')

        # 创建训练和验证数据集
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # batch_size: 每个批次中包含的样本数量。
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)

        # 模型初始化
        if args.model == 'mlp':
            model = MLP(args.input_size, args.output_size, args.hidden_size, args.max_seq_len)
        elif args.model == 'lstm':
            model = LSTMWithMLP(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.mlp_hidden_sizes)
        elif args.model == 'gru':
            model = GRUWithMLP(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.mlp_hidden_sizes)
        elif args.model == 'transformer':
            model = Transformer(args.input_size, args.hidden_size, args.nhead,
                                args.num_layers, args.max_seq_len)

        # 训练
        best_val_loss = train(model, train_loader, val_loader, args, fold, logger)
        results[fold] = best_val_loss
        logger.info(f'Fold {fold + 1} | Best Val Loss: {best_val_loss:.4f}')

    # 输出所有折的平均验证损失
    logger.info(f'K-Fold Cross Validation Results:')
    for fold, val_loss in results.items():
        logger.info(f'Fold {fold + 1} | Val Loss: {val_loss:.4f}')
    logger.info(f'Average Val Loss: {np.mean(list(results.values())):.4f}')

