import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ISIC2016Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, mode='train', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.transform = transform
        self.image_list = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and f.startswith(f'{mode}_')]
        self.image_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '_Segmentation.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

        # 归一化mask为0/1
        mask = (mask > 0.5).float()
        return image, mask

def prepare_dataset(batch_size=4, num_workers=2, image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = ISIC2016Dataset(
        image_dir='data/ISIC2016/images',
        mask_dir='data/ISIC2016/masks',
        mode='train',
        transform=train_transform
    )
    test_dataset = ISIC2016Dataset(
        image_dir='data/ISIC2016/images',
        mask_dir='data/ISIC2016/masks',
        mode='test',
        transform=test_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

if __name__ == '__main__':
    # 测试数据集加载
    train_loader, test_loader = prepare_dataset()
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 测试一个批次
    for images, masks in train_loader:
        print(f"图像形状: {images.shape}")
        print(f"掩码形状: {masks.shape}")
        break 