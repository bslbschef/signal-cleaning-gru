import torch

from losses.hybrid_loss import HybridLoss
from models.mlp import MLP
from models.lstm import LSTM
from models.gru import GRU
from models.transformer import Transformer
from tqdm import tqdm


def test(model, test_loader, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = HybridLoss(args.alpha)

    test_results = []
    with torch.no_grad():
        # for inputs, labels in test_loader:
        for idx, (inputs, labels) in tqdm(enumerate(test_loader), desc=f'Testing: '):
            inputs, labels = inputs[:,:,:3].to(device), labels[:,2].unsqueeze(1).to(device)

            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels).item()

            # 反标准化
            inputs_certer = inputs[:,inputs.shape[1]//2,:]
            raw_input = inputs_certer * test_loader.dataset.input_std[:3].to(device) + test_loader.dataset.input_mean[:3].to(device)
            modification = outputs * test_loader.dataset.label_std[2].to(device) + test_loader.dataset.label_mean[2].to(device)
            raw_label = labels * test_loader.dataset.label_std[2].to(device) + test_loader.dataset.label_mean[2].to(device)

            # 去除填充数据
            sequence_size = inputs.shape[0]
            for i in range(sequence_size):
                test_results.append({
                    'raw_input': raw_input[i,:].cpu().numpy(),
                    'modification': modification[i,:].cpu().numpy(),
                    'raw_label': raw_label[i,:].cpu().numpy(),
                    'loss': loss
                })

    return test_results
