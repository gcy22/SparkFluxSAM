
from skimage import feature
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SAM_utils.build_sam import build_sam_vit_b
from SAM_utils.build_sam import sam_model_registry

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride)
        self.gn1 = nn.GroupNorm(8, out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride),
                nn.GroupNorm(8, out_c)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.relu(out)
        return out

class SparkNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), 
            nn.GroupNorm(4, 32),
            nn.ReLU()
        ) 
        self.layer1 = ResBlock(32, 64, stride=2)

        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=2, dilation=2),
                                    nn.GroupNorm(16, 128), nn.ReLU())

        self.heatmap_head = nn.Conv2d(128, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.prompt_proj = nn.Linear(256, 128)

    def forward(self, x):

        x = self.stem(x)    # [256, 256]
        x = self.layer1(x)  # [128, 128]
        x = self.layer2(x)
        feat = self.layer3(x)

        heatmap_logits = self.heatmap_head(feat)

        b, c, _, _ = feat.shape
        avg_feat = self.avg_pool(feat).flatten(1)
        max_feat = self.max_pool(feat).flatten(1)
        concat_feat = torch.cat([avg_feat, max_feat], dim=1)    # [B, 256]
        prompt_feat = self.prompt_proj(concat_feat)     # [B, 128]

        return heatmap_logits, prompt_feat

class FluxCore(nn.Module):
    def __init__(self, in_dim=256, bottle_dim=64, prompt_dim=128):
        super().__init__()
        self.down_proj = nn.Linear(in_dim, bottle_dim)
        
        self.up_proj = nn.Linear(bottle_dim, in_dim)
        
        self.hyper_net = nn.Sequential(
            nn.Linear(prompt_dim, bottle_dim),
            nn.ReLU(),
            nn.Linear(bottle_dim, bottle_dim) 
        )
        
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, prompt_feat):
        B, C, H, W = x.shape

        shortcut = x
        x = x.permute(0, 2, 3, 1).view(B, -1, C)
        
        x_down = self.down_proj(x)

        dynamic_weight = self.hyper_net(prompt_feat).unsqueeze(1) # [B, 1, bottle_dim]

        x_modulated = x_down * dynamic_weight 
        x_modulated = self.act(x_modulated)
        
        x_up = self.up_proj(x_modulated) # [B, H * W, 256]

        x_up = x_up.view(B, H, W, C).permute(0, 3, 1, 2)

        out = shortcut + self.scale * x_up
        
        return out

def heatmap_to_prompts(logits, original_size=1024, threshold=0.5, training=True):
    probs = torch.sigmoid(logits) # [B, 1, 64, 64]
    batch_size = probs.shape[0]
    device = probs.device
    
    boxes = []
    points = []
    labels = []
    
    scale = original_size / 64.0
    noise_std = 5.0 if training else 0.0

    for i in range(batch_size):
        prob_map = probs[i, 0]
        mask = (prob_map > threshold).float()
        flat_map = prob_map.flatten()
        
        if mask.sum() == 0:
            # 全图框
            box = torch.tensor([0.0, 0.0, float(original_size), float(original_size)]).to(device)
            pos_point = torch.tensor([original_size/2, original_size/2]).to(device)
            neg_point = torch.tensor([0.0, 0.0]).to(device)
        else:
            indices = torch.where(mask > 0)
            # 获取坐标
            y_min, y_max = indices[0].min(), indices[0].max()
            x_min, x_max = indices[1].min(), indices[1].max()
            
            x1 = x_min.item() * scale
            y1 = y_min.item() * scale
            x2 = x_max.item() * scale
            y2 = y_max.item() * scale

            if training:
                noise = lambda: torch.randn(1).item() * noise_std
                x1 = np.clip(x1 + noise(), 0, original_size)
                y1 = np.clip(y1 + noise(), 0, original_size)
                x2 = np.clip(x2 + noise(), 0, original_size)
                y2 = np.clip(y2 + noise(), 0, original_size)
            
            box = torch.tensor([x1, y1, x2, y2]).float().to(device)

            max_idx = torch.argmax(flat_map).item()
            p_y = (max_idx // 64)
            p_x = (max_idx % 64)
            pos_point = torch.tensor([p_x * scale, p_y * scale]).float().to(device)
            
            min_idx = torch.argmin(flat_map).item()
            n_y = (min_idx // 64)
            n_x = (min_idx % 64)
            neg_point = torch.tensor([n_x * scale, n_y * scale]).float().to(device)

        boxes.append(box)
        current_points = torch.stack([pos_point, neg_point], dim=0)
        points.append(current_points)
        current_labels = torch.tensor([1, 0]).to(device)
        labels.append(current_labels)

    box_tensor = torch.stack(boxes).unsqueeze(1)
    point_coords = torch.stack(points)
    point_labels = torch.stack(labels)
    
    return box_tensor, (point_coords, point_labels)

class SparkFluxSAM(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()

        print(f"Loading SAM from {checkpoint_path}...")
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        
        for param in self.sam.parameters():
            param.requires_grad = False

        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
            
        self.SparkNet = SparkNet()
        self.FluxCore = FluxCore(in_dim=256, bottle_dim=64, prompt_dim=128)
        
    def forward(self, image_embeddings, image_small, gt_mask_small=None):    
        # --- Stage 1: SparkNet ---
        SparkNet_logits, prompt_feat = self.SparkNet(image_small) 

        loss_prompt = 0
        if gt_mask_small is not None:
            if SparkNet_logits.shape[-2:] != gt_mask_small.shape[-2:]:
                gt_mask_small = F.interpolate(
                    gt_mask_small.float(), 
                    size=SparkNet_logits.shape[-2:], 
                    mode='nearest' 
                )
            loss_prompt = F.binary_cross_entropy_with_logits(SparkNet_logits, gt_mask_small)

        with torch.no_grad():
            box_prompts, point_prompts = heatmap_to_prompts(SparkNet_logits, 
                                                            original_size=1024,
                                                            training=self.training)
            
        # --- Stage 2: FluxCore ---
        adapted_embeddings = self.FluxCore(image_embeddings, prompt_feat)
            
        # --- Stage 3: SAM Decoder ---
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=point_prompts, # 传入点提示 (coords, labels)
            boxes=box_prompts,    # 传入框提示
            masks=None,
        )
        
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=adapted_embeddings, 
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        return low_res_masks, iou_predictions, loss_prompt