import os
import zipfile
import shutil
from tqdm import tqdm

def extract_zip(zip_path, extract_path):
    """解压zip文件到指定目录"""
    print(f"正在解压 {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"解压完成: {zip_path}")

def organize_dataset():
    """整理数据集目录结构"""
    # 创建必要的目录
    os.makedirs('data/ISIC2016/images', exist_ok=True)
    os.makedirs('data/ISIC2016/masks', exist_ok=True)
    
    # 解压训练数据
    extract_zip('data/ISBI2016_ISIC_Part1_Training_Data.zip', 'data/ISIC2016/temp')
    extract_zip('data/ISBI2016_ISIC_Part1_Training_GroundTruth.zip', 'data/ISIC2016/temp')
    
    # 解压测试数据
    extract_zip('data/ISBI2016_ISIC_Part1_Test_Data.zip', 'data/ISIC2016/temp')
    extract_zip('data/ISBI2016_ISIC_Part1_Test_GroundTruth.zip', 'data/ISIC2016/temp')
    
    # 移动文件到正确的位置
    print("正在整理文件...")
    
    # 移动训练图像
    train_images = [f for f in os.listdir('data/ISIC2016/temp/ISBI2016_ISIC_Part1_Training_Data') 
                   if f.endswith('.jpg')]
    for img in tqdm(train_images, desc="移动训练图像"):
        src = os.path.join('data/ISIC2016/temp/ISBI2016_ISIC_Part1_Training_Data', img)
        dst = os.path.join('data/ISIC2016/images', f'train_{img}')
        shutil.copy2(src, dst)
    
    # 移动训练掩码
    train_masks = [f for f in os.listdir('data/ISIC2016/temp/ISBI2016_ISIC_Part1_Training_GroundTruth') 
                  if f.endswith('.png')]
    for mask in tqdm(train_masks, desc="移动训练掩码"):
        src = os.path.join('data/ISIC2016/temp/ISBI2016_ISIC_Part1_Training_GroundTruth', mask)
        dst = os.path.join('data/ISIC2016/masks', f'train_{mask}')
        shutil.copy2(src, dst)
    
    # 移动测试图像
    test_images = [f for f in os.listdir('data/ISIC2016/temp/ISBI2016_ISIC_Part1_Test_Data') 
                  if f.endswith('.jpg')]
    for img in tqdm(test_images, desc="移动测试图像"):
        src = os.path.join('data/ISIC2016/temp/ISBI2016_ISIC_Part1_Test_Data', img)
        dst = os.path.join('data/ISIC2016/images', f'test_{img}')
        shutil.copy2(src, dst)
    
    # 移动测试掩码
    test_masks = [f for f in os.listdir('data/ISIC2016/temp/ISBI2016_ISIC_Part1_Test_GroundTruth') 
                 if f.endswith('.png')]
    for mask in tqdm(test_masks, desc="移动测试掩码"):
        src = os.path.join('data/ISIC2016/temp/ISBI2016_ISIC_Part1_Test_GroundTruth', mask)
        dst = os.path.join('data/ISIC2016/masks', f'test_{mask}')
        shutil.copy2(src, dst)
    
    # 清理临时目录
    print("清理临时文件...")
    shutil.rmtree('data/ISIC2016/temp')
    
    # 统计数据集信息
    train_images = len([f for f in os.listdir('data/ISIC2016/images') if f.startswith('train_')])
    test_images = len([f for f in os.listdir('data/ISIC2016/images') if f.startswith('test_')])
    
    print("\n数据集整理完成！")
    print(f"训练集图像数量: {train_images}")
    print(f"测试集图像数量: {test_images}")
    print("\n数据集目录结构:")
    print("data/ISIC2016/")
    print("├── images/")
    print("│   ├── train_*.jpg")
    print("│   └── test_*.jpg")
    print("└── masks/")
    print("    ├── train_*.png")
    print("    └── test_*.png")

if __name__ == '__main__':
    organize_dataset() 