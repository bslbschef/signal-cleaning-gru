import os
import torch
import torch.optim as optim
from tqdm import tqdm

from losses.hybrid_loss import HybridLoss
from models.lstm import LSTM
from models.gru import GRU
from models.transformer import Transformer
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(model, train_loader, val_loader, args, fold, logger, checkpoint=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = HybridLoss(args.alpha)

    # setting learning rate scheduler!
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, verbose=True)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.cos_T_max, eta_min=args.cos_eta_min)
    else:
        scheduler = None

    # resuming the training!
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
        best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Resuming training from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")   

    for epoch in tqdm(range(start_epoch, args.epochs), desc=f'Training Fold {fold + 1}'):
        # model training!
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1:03d}'):
            inputs, labels = torch.cat([inputs[:,:,:3],inputs[:,:,-6:]], dim=2).to(device), labels[:,2].unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # model validation!
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = torch.cat([inputs[:,:,:3],inputs[:,:,-6:]], dim=2).to(device), labels[:,2].unsqueeze(1).to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logger.info(f'Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

        # update learning rate scheduler!
        if args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler.step(avg_val_loss)      
        elif args.lr_scheduler == 'CosineAnnealingLR':
            scheduler.step()                  

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(args.model_save_dir, f'best_{args.model}_fold_{fold + 1}.pth')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

            # torch.save(model.state_dict(), model_save_path)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, model_save_path)
            
            logger.info(f'Saved best model for fold {fold + 1} with Val Loss: {best_val_loss:.4f}')

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Val Loss', marker='x')
    plt.title(f'Training and Validation Loss - Model: {args.model} - Fold: {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 保存图形
    plot_filename = os.path.join(args.model_save_dir, f'train_val_loss_fold_{fold + 1}.png')
    plt.savefig(plot_filename)
    plt.close()

    return best_val_loss
