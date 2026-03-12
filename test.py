import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.dataset import OnlinePolypDataset
from src.model import SparkFluxSAM
from src.loss import calculate_hd95


CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 8,
    "num_workers": 4,
    "data_root": "./data/kvasir-seg",
    "sam_checkpoint": "sam_vit_b_01ec64.pth",
    "model_checkpoint": "./checkpoints/best_model.pth",
    "save_visualizations": True,
    "vis_dir": "./test_results",
    "num_vis_samples": 10,
}


def calculate_metrics(pred_mask, gt_mask):

    pred_flat = pred_mask.flatten().cpu().numpy()
    gt_flat = gt_mask.flatten().cpu().numpy()
    
    intersection = np.logical_and(pred_flat, gt_flat).sum()
    union = np.logical_or(pred_flat, gt_flat).sum()
    
    dice = (2.0 * intersection + 1e-5) / (pred_flat.sum() + gt_flat.sum() + 1e-5)
    
    iou = (intersection + 1e-5) / (union + 1e-5)
    
    accuracy = accuracy_score(gt_flat, pred_flat)
    
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    
    tn = np.logical_and(pred_flat == 0, gt_flat == 0).sum()
    fp = np.logical_and(pred_flat == 1, gt_flat == 0).sum()
    specificity = (tn + 1e-5) / (tn + fp + 1e-5) if (tn + fp) > 0 else 0.0
    
    return {
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity
    }

def visualize_prediction(image, pred_mask, gt_mask, save_path):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.squeeze().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.squeeze().cpu().numpy()
    
    image = np.clip(image, 0, 1)
    pred_mask = np.clip(pred_mask, 0, 1)
    gt_mask = np.clip(gt_mask, 0, 1)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    overlay = image.copy()
    overlay[pred_mask > 0.5] = [1, 0, 0]
    overlay[np.logical_and(pred_mask > 0.5, gt_mask > 0.5)] = [0, 1, 0]
    overlay[np.logical_and(pred_mask <= 0.5, gt_mask > 0.5)] = [0, 0, 1]
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Green:TP, Red:FP, Blue:FN)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def test_model():
    print("=" * 60)
    print("开始测试模型性能")
    print("=" * 60)
    
    # 创建保存目录
    if CONFIG["save_visualizations"]:
        os.makedirs(CONFIG["vis_dir"], exist_ok=True)
    
    # 加载测试数据集
    print(f"\n加载测试数据集: {CONFIG['data_root']}")
    
    # 检查数据路径是否存在
    if not os.path.exists(CONFIG["data_root"]):
        print(f"✗ 错误: 数据路径不存在: {CONFIG['data_root']}")
        print(f"   请检查路径是否正确，或修改 CONFIG['data_root']")
        return
    
    test_dataset = OnlinePolypDataset(
        root_dir=CONFIG["data_root"],
        list_name='test',
        training=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )
    print(f"测试集大小: {len(test_dataset)} 张图片")

    print(f"\n加载模型权重: {CONFIG['model_checkpoint']}")
    model = SparkFluxSAM(CONFIG["sam_checkpoint"]).to(CONFIG["device"])
    
    if os.path.exists(CONFIG["model_checkpoint"]):
        checkpoint = torch.load(CONFIG["model_checkpoint"], map_location=CONFIG["device"])
        model.load_state_dict(checkpoint)
        print("✓ 模型权重加载成功")
    else:
        print(f"✗ 错误: 找不到权重文件 {CONFIG['model_checkpoint']}")
        return
    
    model.eval()
    
    all_metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'specificity': [],
        'hd95': []
    }
    
    vis_count = 0
    
    print("\n开始推理...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(CONFIG["device"])
            masks = masks.to(CONFIG["device"])
            
            sam_input = F.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False)
            image_embeddings = model.sam.image_encoder(sam_input)
            
            adapter_input = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
            pred_masks, _, _ = model(image_embeddings, adapter_input, None)
            
            if pred_masks.shape[-2:] != masks.shape[-2:]:
                pred_masks = F.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            pred_probs = torch.sigmoid(pred_masks)
            pred_binary = (pred_probs > 0.5).float()
            
            batch_size = pred_binary.shape[0]
            for i in range(batch_size):
                pred_mask = pred_binary[i, 0]  # [H, W]
                gt_mask = masks[i, 0]  # [H, W]
                
                metrics = calculate_metrics(pred_mask, gt_mask)
                
                pred_mask_np = pred_mask.cpu().numpy()
                gt_mask_np = gt_mask.cpu().numpy()
                hd95_value = calculate_hd95(pred_mask_np, gt_mask_np)
                metrics['hd95'] = hd95_value
                
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
                
                if CONFIG["save_visualizations"] and vis_count < CONFIG["num_vis_samples"]:
                    save_path = os.path.join(
                        CONFIG["vis_dir"],
                        f"sample_{vis_count:03d}_dice_{metrics['dice']:.3f}_iou_{metrics['iou']:.3f}_hd95_{metrics['hd95']:.1f}.png"
                    )
                    visualize_prediction(images[i], pred_binary[i], masks[i], save_path)
                    vis_count += 1
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    mean_metrics = {}
    std_metrics = {}
    for key in all_metrics:
        values = np.array(all_metrics[key])
        mean_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)
    
    print(f"\n测试样本数: {len(all_metrics['dice'])}")
    print("\n指标详情:")
    print(f"  Dice Coefficient:     {mean_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    print(f"  IoU (Jaccard):        {mean_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    print(f"  HD95:                 {mean_metrics['hd95']:.2f} ± {std_metrics['hd95']:.2f}")
    print(f"  Accuracy:             {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"  Precision:            {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"  Recall (Sensitivity): {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"  Specificity:          {mean_metrics['specificity']:.4f} ± {std_metrics['specificity']:.4f}")
    
    result_file = os.path.join(CONFIG["vis_dir"], "test_results.txt")
    with open(result_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("模型测试结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型权重: {CONFIG['model_checkpoint']}\n")
        f.write(f"测试样本数: {len(all_metrics['dice'])}\n\n")
        f.write("指标详情:\n")
        for key in mean_metrics:
            if key == 'hd95':
                f.write(f"  {key.upper()}: {mean_metrics[key]:.2f} ± {std_metrics[key]:.2f}\n")
            else:
                f.write(f"  {key.capitalize()}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n")
    
    print(f"\n✓ 结果已保存到: {result_file}")
    if CONFIG["save_visualizations"]:
        print(f"✓ 可视化结果已保存到: {CONFIG['vis_dir']}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_model()

