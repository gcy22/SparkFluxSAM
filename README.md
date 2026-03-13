## SparkFluxSAM: Fine-tuning SAM for 2D Endoscopic Segmentation

This repository implements the `SparkFluxSAM` model built on top of the Segment Anything Model (SAM) architecture for 2D endoscopic / polyp segmentation (the current example uses the `Kvasir-SEG` dataset). The project freezes the SAM image encoder and adds lightweight adaptation and prompt-generation modules to efficiently transfer SAM to medical image segmentation tasks.

---

### Key Features

- **SAM fine-tuning**: Uses the official SAM ViT-B backbone, with the image encoder frozen by default while fine-tuning the mask decoder and custom adaptation modules.
- **SparkNet for prompt generation**: Learns to generate heatmaps and point/box prompts from downstream task images to guide SAM to produce more accurate masks.
- **FluxCore adaptation module**: Applies bottleneck projection and a hyper-network to dynamically modulate SAM image features, improving adaptation to target shapes and boundaries.
- **Structured loss function**: Combines Dice, Focal, and BCE in `StructureLoss` to jointly optimize region overlap and hard examples.
- **End-to-end training & evaluation**: Supports training, validation, testing, and visualization with metrics including Dice, IoU, HD95, Accuracy, Precision, Recall, and Specificity.

---

### Project Structure

```text
code/
├─ train.py                # Training script (warmup + cosine scheduler, grad accumulation, etc.)
├─ test.py                 # Testing and visualization script
├─ split_data.py           # (Optional) dataset splitting script
├─ requirements.txt        # Python dependencies
├─ src/
│  ├─ model.py             # SparkFluxSAM definition (SparkNet + FluxCore + SAM)
│  ├─ dataset.py           # OnlinePolypDataset definition
│  └─ loss.py              # StructureLoss and Dice/IoU/HD95 metrics
├─ SAM_utils/
│  ├─ build_sam.py         # SAM builders (ViT-B/L/H)
│  └─ transforms.py        # (Optional) SAM-related image transforms
├─ SAM_model/
│  ├─ image_encoder.py     # SAM image encoder
│  ├─ mask_decoder.py      # SAM mask decoder
│  ├─ prompt_encoder.py    # SAM prompt encoder
│  ├─ transformer.py       # SAM two-way transformer
│  └─ sam.py               # SAM main wrapper
└─ .idea/                  # IDE configuration (can be ignored)
```

---

### Environment Setup

#### 1. Clone the repository

```bash
git clone <your_repo_url>
cd code
```

#### 2. Create environment and install dependencies

It is recommended to use Python 3.9/3.10 and install a CUDA-compatible version of PyTorch.

```bash
pip install -r requirements.txt
```

`requirements.txt` mainly includes:

- `torch`, `torchvision`
- `numpy`, `scipy`
- `opencv-python-headless`
- `albumentations`
- `tqdm`, `matplotlib`
- `scikit-learn`
- `protobuf<3.21.0`

---

### SAM Pretrained Weights

This project uses SAM ViT-B weights by default:

- In `train.py` and `test.py`, the checkpoint is specified via `CONFIG["sam_checkpoint"]`:
  - Default value: `sam_vit_b_01ec64.pth`
- Please download the **SAM ViT-B checkpoint file** (e.g., `sam_vit_b_01ec64.pth`) from the **official SAM GitHub repository** and place it in the project root (or another path of your choice), then update the config accordingly, e.g.:
  - `CONFIG["sam_checkpoint"] = "path/to/sam_vit_b_01ec64.pth"`

The SAM builder is implemented in `SAM_utils/build_sam.py`, where `sam_model_registry["vit_b"]` loads the model with the given checkpoint.

---

### Dataset Preparation

The current code assumes a dataset layout similar to `Kvasir-SEG`:

```text
data/
└─ kvasir-seg/
   ├─ images/          # RGB images (.jpg or .png)
   ├─ masks/           # binary masks (.jpg or .png)
   ├─ train.txt        # training image names (without extension)
   ├─ val.txt          # validation image names
   └─ test.txt         # test image names
```

- Each line in `train.txt / val.txt / test.txt` is an image name, e.g.:

  ```text
  0001
  0002
  0003
  ...
  ```

The dataset root can be modified via `CONFIG["data_root"]` in `train.py` and `test.py`:

```python
"data_root": "./data/kvasir-seg"
```

#### Data augmentation & preprocessing

- `OnlinePolypDataset` in `src/dataset.py` handles loading images and masks. By default, it:
  - Resizes images to \(512 \times 512\)
  - Applies `albumentations` transforms (random flips, scale/rotate, brightness/contrast, noise, etc.)
  - Performs ImageNet-style normalization
- `EnhancedDataset` in `train.py` uses stronger augmentations (`get_strong_augmentation()`) during training to improve generalization.

> If you have limited GPU memory (e.g., RTX 4060), you can reduce the image size to \(256 \times 256\) and adjust `batch_size` and `accum_iter` accordingly (notes are already included in the code).

---

### Model Overview

The core model is defined in `src/model.py` as `SparkFluxSAM`:

- **SAM backbone**
  - Uses `sam_model_registry["vit_b"]` to build the SAM ViT-B model.
  - By default, **all SAM parameters are frozen**, except for the `mask_decoder` (`requires_grad=True`).
- **SparkNet**
  - A lightweight CNN module:
    - Input: downsampled image (default \(256 \times 256\))
    - Output:
      - Low-resolution heatmap `heatmap_logits` indicating potential target regions
      - Prompt feature vector `prompt_feat` used for feature modulation
- **Heatmap-to-Prompts**
  - The `heatmap_to_prompts` function:
    - Converts SparkNet heatmaps into:
      - Box prompts `box_prompts`
      - Point prompts `point_prompts` (positive/negative points)
    - Adds random noise during training to improve robustness and diversity.
- **FluxCore**
  - A bottleneck adaptation module:
    - Flattens SAM image encoder features into sequences
    - Projects to a lower-dimensional `bottle_dim`, modulates with `prompt_feat` via a hyper-network
    - Projects back to the original feature dimension and adds a residual connection
- **SAM Prompt & Mask Decoder**
  - Uses the original `PromptEncoder` to embed box and point prompts into sparse/dense prompt embeddings.
  - Uses `MaskDecoder` to output low-resolution masks `low_res_masks` and IoU predictions `iou_predictions`.

The forward interface is:

```python
low_res_masks, iou_predictions, loss_prompt = model(image_embeddings, image_small, gt_mask_small=None)
```

- `image_embeddings`: features from the SAM image encoder at \(1024 \times 1024\)
- `image_small`: downsampled input image (default \(256 \times 256\)) used by SparkNet
- `gt_mask_small`: downsampled mask (only needed during training to compute `loss_prompt`)

---

### Loss Function and Metrics

Implemented in `src/loss.py`:

- **StructureLoss**
  - The total loss is:


$\mathcal{L} = w_{\text{dice}} \cdot \mathcal{L}_{\text{dice}} + w_{\text{focal}} \cdot \mathcal{L}_{\text{focal}} + w_{\text{bce}} \cdot \mathcal{L}_{\text{bce}}$

  - Where:
    - Dice loss: improves region overlap quality
    - Focal loss: focuses on hard examples and alleviates class imbalance
    - BCE loss: stabilizes training

- **Metrics**
  - `calculate_dice`: Dice coefficient
  - `calculate_iou`: IoU (Jaccard)
  - `calculate_hd95`: 95th percentile Hausdorff distance based on mask contours and distance transforms

During validation in `train.py`, the script reports `Loss / Dice / IoU / HD95`. In `test.py`, additional metrics such as Accuracy, Precision, Recall, and Specificity are computed and summarized.

---

### Experimental Results

The following table reports quantitative results on the target dataset, comparing `SparkFluxSAM` (Ours) with several representative segmentation baselines:

| Method    | Params (M) | DSC ↑  | IoU ↑  | HD95 (mm) ↓ |
|----------|------------|-------:|-------:|------------:|
| U-Net    | 33.22      | 0.8567 | 0.6930 | 16.47       |
| TransUNet| 107.68     | 0.8738 | 0.7825 | 14.35       |
| HResFormer | 160.39   | 0.8523 | 0.7494 | **12.27**   |
| VM-UNet  | 181.46     | 0.8744 | 0.7505 | 14.21       |
| SAM      | -          | 0.7521 | 0.7360 | 22.68       |
| **Ours (SparkFluxSAM)** | 95.40 | **0.8895** | **0.7995** | 12.30 |

`SparkFluxSAM` achieves the best DSC and IoU with a moderate parameter count, while maintaining competitive HD95 compared to the strongest baseline.

---

### Training

Run the following command from the `code` directory:

```bash
python train.py
```

Key fields in the `CONFIG` dictionary in `train.py`:

- **Basic training settings**
  - `lr`: initial learning rate (default `1e-4`)
  - `min_lr`: minimum learning rate for cosine decay
  - `warmup_epochs`: number of warmup epochs
  - `epochs`: total number of training epochs
  - `batch_size`: batch size (reduce if GPU memory is limited)
  - `accum_iter`: gradient accumulation steps (to simulate larger batches)
  - `num_workers`: DataLoader workers
  - `device`: training device (auto-selects CUDA if available)
  - `save_dir`: directory to save model checkpoints (default `./checkpoints`)

- **Optimization and regularization**
  - Optimizer: `AdamW` (`weight_decay=1e-4`, `betas=(0.9, 0.999)`)
  - Scheduler: `WarmupCosineScheduler`
  - Gradient clipping: `grad_clip=1.0`
  - AMP: `torch.cuda.amp` mixed precision

- **SAM-related options**
  - `sam_checkpoint`: path to SAM checkpoint
  - `unfreeze_sam_layers`: whether to unfreeze part of the SAM image encoder (e.g., last `LayerNorm` layers)

During training:

- Metrics are evaluated on the validation set every epoch
- Checkpoints are saved automatically:
  - `best_model.pth`: best model according to validation Dice
  - `last_model.pth`: last epoch model

---

### Testing and Visualization

After training, you can evaluate the model and generate visualizations using `test.py`:

```bash
python test.py
```

The relevant fields in the `CONFIG` dictionary in `test.py` include:

- `data_root`: test dataset root (same as training)
- `sam_checkpoint`: SAM checkpoint path
- `model_checkpoint`: trained model path (default `./checkpoints/best_model.pth`)
- `save_visualizations`: whether to save visualization images
- `vis_dir`: directory to save visualizations (default `./test_results`)
- `num_vis_samples`: number of samples to visualize

The test script will:

- Load the test set using `OnlinePolypDataset`
- Compute per-image metrics
- Aggregate and print mean ± std for all metrics
- Save a textual summary to `test_results.txt`
- Optionally save visualization images (original, GT, prediction, and TP/FP/FN overlay)

---

### Example Workflow (Train & Test)

1. **Prepare the dataset**, ensuring the directory structure and `*.txt` split files are correct.
2. **Download SAM ViT-B weights** from the official SAM GitHub repository and set `CONFIG["sam_checkpoint"]` to the actual path.
3. (Optional) Adjust for GPU memory:
   - Image size (Resize in `dataset.py` / `train.py`)
   - `batch_size` and `accum_iter`
4. Run training from the `code` directory:

   ```bash
   python train.py
   ```

5. After training, evaluate using the best checkpoint:

   ```bash
   python test.py
   ```

6. Inspect:
   - Metrics printed in the terminal
   - `test_results/test_results.txt`
   - Visualization images in `test_results/`

---

### Acknowledgements

This project is built upon Meta's **Segment Anything Model (SAM)** architecture. Copyright and licensing follow
the official SAM repository `LICENSE`.  
If you use this code or methodology in your work, please cite the original SAM paper and the corresponding paper/report for this project.

---

### TODO / Future Work

- Support more medical image datasets (e.g., CVC-ClinicDB)
- Add support for larger SAM backbones such as ViT-L / ViT-H
- Extend to multi-class and multi-organ segmentation
- Explore alternative prompt generation strategies and multi-scale fusion structures



