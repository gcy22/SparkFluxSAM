import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from torch.utils.data import Dataset

class OnlinePolypDataset(Dataset):
    def __init__(self, root_dir, list_name, training=True):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.training = training
        
        list_path = os.path.join(root_dir, f'{list_name}.txt')
        with open(list_path, 'r') as f:
            self.names = [line.strip() for line in f.readlines()]

        # 512x512
        if self.training:
            self.transform = A.Compose([
                A.Resize(512, 512),     # 4060 如果跑不起来要调整这个，改成256 * 256
                
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),

                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(512, 512), # 如果上面改成256了这里也要改成 256
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.image_dir, name + '.jpg')
        if not os.path.exists(img_path): img_path = img_path.replace('.jpg', '.png')
        
        mask_path = os.path.join(self.mask_dir, name + '.jpg')
        if not os.path.exists(mask_path): mask_path = mask_path.replace('.jpg', '.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented['image']
        mask_tensor = augmented['mask']

        mask_tensor = mask_tensor.float().unsqueeze(0) / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor