import torch
import torch.nn as nn
import torch.fft


class HybridLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        # reduction='none'：表示不进行任何聚合操作，返回每个元素的损失值。输出的形状与输入的形状相同。
        # reduction='mean'：表示对所有元素的损失值取平均，返回一个标量。
        # reduction='sum'：表示对所有元素的损失值求和，返回一个标量。
        # self.mse = nn.MSELoss(reduction='none')
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        # 时域损失
        time_loss = self.mse(pred, target)

        # 频域损失
        # pred_fft = torch.fft.fft(pred, dim=1)
        # target_fft = torch.fft.fft(target, dim=1)
        # freq_loss = self.mse(torch.abs(pred_fft), torch.abs(target_fft))

        # return self.alpha * time_loss + (1 - self.alpha) * freq_loss
        # return freq_loss
        return time_loss

