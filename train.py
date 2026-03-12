
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import SparkFluxSAM
from src.loss import StructureLoss, calculate_dice, calculate_iou, calculate_hd95

# ==========================================
# 如果要用4060训练的话，请设置: batch_size = 1; accum_iter = 8/16
# ==========================================
CONFIG = {
    "lr": 1e-4,
    "min_lr": 5e-6,
    "warmup_epochs": 5,
    "batch_size": 8,
    "accum_iter": 2,
    "epochs": 100,
    "num_workers": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "weight_decay": 1e-4,
    "grad_clip": 1.0,         # 梯度裁剪
    "save_dir": "./checkpoints",
    "data_root": "./data/kvasir-seg",
    "sam_checkpoint": "sam_vit_b_01ec64.pth",
    "unfreeze_sam_layers": False,  # 是否解冻 SAM 的部分层（需要更多显存）
    "use_reduce_lr": True,
    "patience": 10,
}

warnings.filterwarnings("ignore", message="Error fetching version info")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

# ==========================================
# 数据增强策略
# ==========================================
def get_strong_augmentation():
    return A.Compose([
        A.Resize(512, 512),

        # 几何变换（增强）
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),  # 增加旋转角度
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=10,
            p=0.5
        ),
        # ----------- 用于提升性能的更高级预处理 -------------- #
        # A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
        # A.GridDistortion(p=0.3),

        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.2),
        # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        # A.GaussianBlur(blur_limit=3, p=0.2),
        # A.MedianBlur(blur_limit=3, p=0.2),

        # 归一化
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class OnlinePolypDataset(Dataset):
    def __init__(self, root_dir, list_name, training=True):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.training = training

        list_path = os.path.join(root_dir, f'{list_name}.txt')
        with open(list_path, 'r') as f:
            self.names = [line.strip() for line in f.readlines()]

        if self.training:
            self.transform = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(512, 512),
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

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    val_dice = 0
    val_iou = 0  # 新增 IoU
    val_hd95 = 0
    num_samples = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)

            # SAM Embeddings
            sam_input = F.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False)
            image_embeddings = model.sam.image_encoder(sam_input)

            # Forward
            pred_masks, _, _ = model(image_embeddings, images, None)

            if pred_masks.shape[-2:] != masks.shape[-2:]:
                pred_masks = F.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            # Loss
            loss = criterion(pred_masks, masks)
            val_loss += loss.item()

            # 准备 Binary Mask 用于指标计算
            pred_sigmoid = torch.sigmoid(pred_masks)
            pred_binary = (pred_sigmoid > 0.5).float()

            # --- 调用 src.loss 中的函数 ---
            # 1. Dice
            val_dice += calculate_dice(pred_binary, masks)

            # 2. IoU
            val_iou += calculate_iou(pred_binary, masks)

            # 3. HD95 (需要转 numpy, 逐样本计算)
            batch_size = pred_binary.shape[0]
            num_samples += batch_size

            pred_np = pred_binary.cpu().numpy() # [B, 1, H, W]
            gt_np = masks.cpu().numpy()         # [B, 1, H, W]

            for i in range(batch_size):
                # 取出单张图片的 [H, W] 维度
                hd95_val = calculate_hd95(pred_np[i, 0], gt_np[i, 0])
                val_hd95 += hd95_val

    # 计算平均值
    return (val_loss / len(loader),
            val_dice / len(loader),
            val_iou / len(loader),
            val_hd95 / num_samples)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=5e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    print("=" * 60)
    print("优化训练配置")
    print("=" * 60)
    print(f"初始学习率: {CONFIG['lr']}")
    print(f"Warmup 轮数: {CONFIG['warmup_epochs']}")
    print(f"梯度裁剪: {CONFIG['grad_clip']}")
    print(f"增强数据增强: 启用")
    print("=" * 60)

    # Dataset with enhanced augmentation
    class EnhancedDataset(OnlinePolypDataset):
        def __init__(self, root_dir, list_name, training=True):
            super().__init__(root_dir, list_name, training=False)  # 先调用父类
            self.training = training
            if training:
                self.transform = get_strong_augmentation()
            else:
                self.transform = A.Compose([
                    A.Resize(512, 512),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])

    print("初始化数据集...")
    train_ds = EnhancedDataset(CONFIG["data_root"], 'train', training=True)
    val_ds = EnhancedDataset(CONFIG["data_root"], 'val', training=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=CONFIG["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=CONFIG["num_workers"],
                            pin_memory=True)

    # Model
    model = SparkFluxSAM(CONFIG["sam_checkpoint"]).to(CONFIG["device"])
    for param in model.sam.image_encoder.parameters():
        param.requires_grad = False

    if CONFIG["unfreeze_sam_layers"]:
        print("解冻 SAM 的部分层...")
        # 解冻最后几层的 LayerNorm
        for name, param in model.sam.image_encoder.named_parameters():
            if "norm" in name and any(x in name for x in ["blocks.23", "blocks.22", "blocks.21"]):
                param.requires_grad = True
                print(f"  解冻: {name}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999)  # 使用默认 beta
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=CONFIG["warmup_epochs"],
        total_epochs=CONFIG["epochs"],
        min_lr=1e-6
    )

    criterion = StructureLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_dice = 0.0
    best_epoch = 0

    print("\n开始训练...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        if epoch < 5:
            lambda_prompt = 2.0; lambda_iou = 0.1
        else:
            lambda_prompt = 0.5
            lambda_iou = 2.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Ep {epoch +1}")

        for i, (images, masks) in pbar:
            images = images.to(CONFIG["device"])
            masks = masks.to(CONFIG["device"])

            with torch.cuda.amp.autocast():
                sam_input = F.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    image_embeddings = model.sam.image_encoder(sam_input)

                adapter_input = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
                pred_masks, iou_pred, loss_prompt = model(image_embeddings, adapter_input, masks)

                if pred_masks.shape[-2:] != masks.shape[-2:]:
                    pred_masks = F.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                loss_seg = criterion(pred_masks, masks)

                # IoU Loss
                with torch.no_grad():
                    pred_binary = (pred_masks > 0).float()
                    intersection = (pred_binary * masks).sum(dim=(2, 3))
                    union = pred_binary.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                    gt_iou = intersection / (union + 1e-5)
                    gt_iou = gt_iou.unsqueeze(1)

                loss_iou = F.mse_loss(iou_pred, gt_iou)
                total_loss = (loss_seg + lambda_prompt * loss_prompt + lambda_iou * loss_iou) / CONFIG["accum_iter"]

            scaler.scale(total_loss).backward()

            if (i + 1) % CONFIG["accum_iter"] == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_loss = total_loss.item() * CONFIG["accum_iter"]
            train_loss += current_loss

            current_lr = scheduler.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f"{current_loss:.3f}",
                'Seg': f"{loss_seg.item():.3f}",
                'LR': f"{current_lr:.2e}"
            })

        current_lr = scheduler.step(epoch)

        val_loss, val_dice, val_iou, val_hd95 = validate(model, val_loader, criterion, CONFIG["device"])
        print \
            (f"Ep {epoch +1} | LR: {current_lr:.2e} |Dice: {val_dice:.4f}  | Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | HD95: {val_hd95:.2f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
            print(f"🎉 New Best Dice: {best_dice:.4f} (IoU: {val_iou:.4f}, HD95: {val_hd95:.2f}) | Model Saved!")

        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "last_model.pth"))

    print(f"\n训练完成！最佳 Dice: {best_dice:.4f} (Epoch {best_epoch})")

if __name__ == "__main__":
    main()

