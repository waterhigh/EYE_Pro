import os
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AgeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize(1024),  # 先缩放到中等尺寸
            transforms.RandomRotation(10),
            transforms.RandomCrop(800),  # 随机裁剪
            transforms.Resize(256),   # 二次缩放
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.samples = []

        for sample_dir in os.listdir(root_dir):
            sample_path = os.path.join(root_dir, sample_dir)
            if not os.path.isdir(sample_path):
                continue
            
            try:
                age = int(sample_dir.split('_')[0])
            except:
                continue
            
            category_dirs = sorted([d for d in os.listdir(sample_path) 
                                  if os.path.isdir(os.path.join(sample_path, d))])
            if len(category_dirs) != 4:
                continue
            
            img_paths = []
            for cat_dir in category_dirs:
                cat_path = os.path.join(sample_path, cat_dir)
                images = [f for f in os.listdir(cat_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not images:
                    break
                img_paths.append(os.path.join(cat_path, images[0]))
            
            if len(img_paths) == 4:
                self.samples.append((img_paths, age))

    def __len__(self):
        return len(self.samples)

    def optimized_loader(self, path):
        try:
            # 分离文件打开和数据处理
            with open(path, 'rb') as f:
                img = Image.open(f)
                img.draft('RGB', (1024, 768))  # 设置draft模式
                # 将数据完整加载到内存
                img.load()  
                return img.convert('RGB')
        except Exception as e:
            print(f"图像加载失败：{path} - {str(e)}")
            # 生成红色错误提示图
            error_img = Image.new('RGB', (224, 224), (255, 0, 0))
            draw = ImageDraw.Draw(error_img)
            draw.text((10,10), "LOAD ERROR", fill=(255,255,255))
            return error_img

    def __getitem__(self, idx):
            img_paths, age = self.samples[idx]
            images = []
            for path in img_paths:
                img = self.optimized_loader(path)  
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            return (*images, torch.tensor(age, dtype=torch.float32))  # 注意这里的返回格式

def get_dataloader(root_dir, batch_size=1):
    dataset = AgeDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)