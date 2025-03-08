import torch
import torch.nn as nn
import torch.fft


class HybridLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, mask):
        # 扩展mask维度
        mask = mask.unsqueeze(-1).expand_as(pred)
        valid_elements = mask.sum()

        # 时域损失
        time_loss = (self.mse(pred, target) * mask).sum() / valid_elements

        # 频域损失
        pred_fft = torch.fft.fft(pred, dim=1)
        target_fft = torch.fft.fft(target, dim=1)
        freq_loss = (self.mse(torch.abs(pred_fft), torch.abs(target_fft)) * mask).sum() / valid_elements

        return self.alpha * time_loss + (1 - self.alpha) * freq_loss
