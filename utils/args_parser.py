import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, 
                        choices=['mlp', 'lstm', 'gru', 'transformer'],
                        default='lstm')
    parser.add_argument('--nhead', type=int, default=4, help='Transformer model: nhead')
    parser.add_argument('--max_seq_len', type=int, default=20*60*1, help='Maximum sequence length for LSTM/GRU model')

    parser.add_argument('--input_size', type=int, default=1+1)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1)
    # nargs='+': represent that the parameter can accept one or more values and returns a list of these values!
    parser.add_argument('--mlp_hidden_sizes', type=int, nargs='+', default=[64, 32], help='after model: MLP hidden sizes')

    parser.add_argument('--input_dir', type=str, default='./datasets/data_train/inputs')
    parser.add_argument('--label_dir', type=str, default='./datasets/data_train/labels')
    parser.add_argument('--test_input_dir', type=str, default='./datasets/data_test/inputs')
    parser.add_argument('--test_label_dir', type=str, default='./datasets/data_test/labels')
    parser.add_argument('--model_save_dir', type=str, default='./model_checkpoints')  
    parser.add_argument('--logger_dir', type=str, default='./logger')  
    parser.add_argument('--result_dir', type=str, default='./result')  

    parser.add_argument('--batch_size', type=int, default=50*4, help='sample amount per batch')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', help='ReduceLROnPlateau || CosineAnnealingLR')
    parser.add_argument('--cos_T_max', type=int, default=100, help='cosine rl: the epoch number of the cycle')
    parser.add_argument('--cos_eta_min', type=float, default=1e-6, help='cosine rl: minimul learning rate')

    parser.add_argument('--alpha', type=float, default=0.3, help='Weight for time domain loss')
    parser.add_argument('--if_kfold', type=bool, default=False)
    parser.add_argument('--k_folds', type=int, default=5) 
    
    parser.add_argument('--resuming_training', type=bool, default=False)  
    parser.add_argument('--resuming_model_name', type=str, default=None)  
    return parser.parse_args()