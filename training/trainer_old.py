import os
import torch
import torch.optim as optim
from losses.hybrid_loss import HybridLoss
from models.lstm import LSTM
from models.gru import GRU
from models.transformer import Transformer


def train(model, train_loader, val_loader, args, fold, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = HybridLoss(args.alpha)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels, lengths in train_loader:
            inputs, labels, lengths = inputs[:,:,:4].to(device), labels[:,:,:4].to(device), lengths.to(device)
            # 生成mask
            batch_size, seq_len = inputs.shape[:2]
            mask = torch.zeros(batch_size, seq_len, device=device)
            for i in range(batch_size):
                mask[i, :lengths[i]] = 1

            optimizer.zero_grad()

            # 模型前向
            if isinstance(model, Transformer):
                src_key_padding_mask = (mask == 0)
                outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
            elif isinstance(model, (LSTM, GRU)):
                outputs = model(inputs, lengths)
            else:
                outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels, mask)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, lengths in val_loader:
                inputs, labels, lengths = inputs[:,:,:4].to(device), labels[:,:,:4].to(device), lengths.to(device)

                batch_size, seq_len = inputs.shape[:2]
                mask = torch.zeros(batch_size, seq_len, device=device)
                for i in range(batch_size):
                    mask[i, :lengths[i]] = 1

                if isinstance(model, Transformer):
                    src_key_padding_mask = (mask == 0)
                    outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
                elif isinstance(model, (LSTM, GRU)):
                    outputs = model(inputs, lengths)
                else:
                    outputs = model(inputs)

                val_loss += criterion(outputs, labels, mask).item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'Epoch {epoch + 1:03d} | Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(args.model_save_dir, f'best_{args.model}_fold_{fold + 1}.pth')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logger.info(f'Saved best model for fold {fold + 1} with Val Loss: {best_val_loss:.4f}')

    return best_val_loss