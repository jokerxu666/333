import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model import TransUNet
from dataset import prepare_dataset
from config import *
from utils import dice_coefficient, calculate_metrics

def dice_coefficient(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    total_dice = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, masks)
        pbar.set_postfix({'loss': loss.item(), 'dice': dice_coefficient(outputs, masks)})
    return total_loss / len(loader), total_dice / len(loader)

def validate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss = 0
    total_dice = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]')
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks)
            pbar.set_postfix({'loss': loss.item(), 'dice': dice_coefficient(outputs, masks)})
    return total_loss / len(loader), total_dice / len(loader)

def plot_curves(train_losses, val_losses, train_dices, val_dices, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.arange(1, len(train_losses)+1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_dices, label='Train Dice')
    plt.plot(epochs, val_dices, label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Dice Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'dice_curve.png'))
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    train_loader, test_loader = prepare_dataset(batch_size=4, num_workers=2, image_size=224)
    model = TransUNet(in_channels=3, out_channels=1, img_size=224).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 30
    print('Start training...')
    train_losses, val_losses, train_dices, val_dices = [], [], [], []
    for epoch in range(num_epochs):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        val_loss, val_dice = validate(model, test_loader, criterion, device, epoch, num_epochs)
        print(f'Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
    print('训练完成！')
    plot_curves(train_losses, val_losses, train_dices, val_dices, save_dir='results')
    print('Loss和Dice曲线已保存到results目录下！')

if __name__ == '__main__':
    main() 