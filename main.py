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
from models.lstm_certerPoint_multiParallel import LSTMWithMLP
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

    # judge if use k-fold cross validation!
    if args.if_kfold:
        k_folds = args.k_folds
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        splits = kfold.split(dataset)
    else:
        k_folds = 1  
        total_indices = list(range(len(dataset)))
        splits = [(total_indices, total_indices)]  

    # store the total loss for each fold!
    results = {}

    for fold, (train_ids, val_ids) in enumerate(splits):
        logger.info(f'Fold {fold + 1}/{k_folds}')
        logger.info(f'Number of train samples: {len(train_ids)}')
        logger.info(f'Number of val samples: {len(val_ids)}')

        # create samplers of train and validation sets!
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # batch_size: the number of samples in each batch!
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)

        # model initialization!
        if args.model == 'mlp':
            model = MLP(args.input_size, args.output_size, args.hidden_size, args.max_seq_len)
        elif args.model == 'lstm':
            model = LSTMWithMLP(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.mlp_hidden_sizes)
        elif args.model == 'gru':
            model = GRUWithMLP(args.input_size, args.hidden_size, args.num_layers, args.output_size, args.mlp_hidden_sizes)
        elif args.model == 'transformer':
            model = Transformer(args.input_size, args.hidden_size, args.nhead, args.num_layers, args.max_seq_len)

        # check if resuming training!
        if args.resuming_training and args.resuming_model_name:
            model_path = os.path.join(args.model_save_dir, args.resuming_model_name)
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
            else:
                checkpoint = None
                logger.warning(f'Model file not found: {model_path}. Starting training from scratch.')
        else:
            checkpoint = None

        # training!
        best_val_loss = train(model, train_loader, val_loader, args, fold, logger, checkpoint)
        results[fold] = best_val_loss
        logger.info(f'Fold {fold + 1} | Best Val Loss: {best_val_loss:.4f}')

    # logging the individual and averaged validation loss of all folds!
    logger.info(f'K-Fold Cross Validation Results:')
    for fold, val_loss in results.items():
        logger.info(f'Fold {fold + 1} | Val Loss: {val_loss:.4f}')
    logger.info(f'Average Val Loss: {np.mean(list(results.values())):.4f}')

