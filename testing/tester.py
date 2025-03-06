import torch

from losses.hybrid_loss import HybridLoss
from models.mlp import MLP
from models.lstm import LSTM
from models.gru import GRU
from models.transformer import Transformer


def test(model, test_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = HybridLoss(args.alpha)

    test_results = []
    with torch.no_grad():
        # for inputs, labels, lengths in test_loader:
        for idx, (inputs, labels, lengths) in enumerate(test_loader):
            if idx == len(test_loader)-1:
                print(1)
            inputs, labels, lengths = inputs[:,:,:4].to(device), labels[:,:,:4].to(device), lengths.to(device)

            # 生成mask
            batch_size, seq_len = inputs.shape[:2]
            mask = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                mask[i, :lengths[i]] = 1

            # 模型前向
            if isinstance(model, Transformer):
                src_key_padding_mask = (mask == 0)
                outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
            elif isinstance(model, (LSTM, GRU)):
                outputs = model(inputs, lengths)
            else:
                outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels, mask).item()

            # 反标准化
            raw_input = inputs * test_loader.dataset.input_std[:4].to(device) + test_loader.dataset.input_mean[:4].to(device)
            modification = outputs * test_loader.dataset.label_std[:4].to(device) + test_loader.dataset.label_mean[:4].to(device)
            raw_label = labels * test_loader.dataset.label_std[:4].to(device) + test_loader.dataset.label_mean[:4].to(device)

            # 去除填充数据
            for i in range(batch_size):
                length = lengths[i].item()
                test_results.append({
                    'raw_input': raw_input[i,:length].cpu().numpy(),
                    'modification': modification[i,:length].cpu().numpy(),
                    'raw_label': raw_label[i,:length].cpu().numpy(),
                    'length': length,
                    'loss': loss
                })

    return test_results
