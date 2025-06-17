import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import prepare_dataset
from model import TransUNet
from config import Config
from utils import calculate_metrics, visualize_prediction
from PIL import Image

def evaluate_model(model, test_loader, device, save_dir='results'):
    """
    评估模型性能并可视化结果
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 存储所有预测结果
    all_metrics = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            
            # 计算指标
            metrics = calculate_metrics(preds, masks)
            all_metrics.append(metrics)
            
            # 可视化结果
            for j in range(images.size(0)):
                image = images[j].cpu().numpy().transpose(1, 2, 0)
                mask = masks[j].cpu().numpy().squeeze()
                pred = preds[j].cpu().numpy().squeeze()
                
                # 反归一化图像
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = std * image + mean
                image = np.clip(image, 0, 1)
                
                # 保存可视化结果
                save_path = os.path.join(save_dir, f'sample_{i}_{j}.png')
                visualize_prediction(image, mask, pred, save_path)
    
    # 计算平均指标
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics])
    }
    
    # 打印评估结果
    print("\n评估结果:")
    print(f"Dice系数: {avg_metrics['dice']:.4f}")
    print(f"IoU: {avg_metrics['iou']:.4f}")
    print(f"精确率: {avg_metrics['precision']:.4f}")
    print(f"召回率: {avg_metrics['recall']:.4f}")
    
    # 保存评估结果
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write("评估结果:\n")
        f.write(f"Dice系数: {avg_metrics['dice']:.4f}\n")
        f.write(f"IoU: {avg_metrics['iou']:.4f}\n")
        f.write(f"精确率: {avg_metrics['precision']:.4f}\n")
        f.write(f"召回率: {avg_metrics['recall']:.4f}\n")
    
    return avg_metrics

def visualize_prediction(image, mask, pred, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = prepare_dataset(batch_size=1, num_workers=0, image_size=224)
    model = TransUNet(in_channels=3, out_channels=1, img_size=224).to(device)
    # 加载最后一轮权重（如有保存）
    if os.path.exists('checkpoints/best_model.pth'):
        model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device)['model_state_dict'])
    model.eval()
    os.makedirs('results/vis', exist_ok=True)
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc='Visualizing')):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()[0,0]
            pred_mask = (preds > 0.5).astype(np.uint8)
            # 还原原图
            image_np = images.cpu().numpy()[0].transpose(1,2,0)
            image_np = (image_np * 255).astype(np.uint8)
            mask_np = masks.cpu().numpy()[0,0]
            save_path = f'results/vis/sample_{idx}.png'
            visualize_prediction(image_np, mask_np, pred_mask, save_path)
            if idx < 20:
                print(f'Saved: {save_path}')
            if idx >= 49:
                break
    print('可视化结果已保存到 results/vis/')

if __name__ == '__main__':
    main() 