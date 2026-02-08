# Pretrain 3D UNet for tumor segmentation

Steps and commands to pretrain the UNet, then use it (frozen) in CycleGAN.

---

## 1. Prerequisites

- **Data**: Training images in `DATA_PATH/images/` and tumor masks in `MASK_DIR/`. Same number of files in both, same order after sorting (e.g. `001.nii.gz` in images ↔ first mask in `lstFiles(MASK_DIR)`).
- **Patch size**: Use the same as CycleGAN (default `128 128 64`).
- **Optional**: `pip install medpy` for HD95 during training/eval.

---

## 2. Train the UNet (pretrain)

Run from the **project root** (`3D_CycleGAN_Tumor_Aware/`):

```bash
source /home/ashraful/projects/def-kovens/ashraful/3D-CycleGan-Pytorch-MedImaging-attention/env_mr_seq/bin/activate

cd /home/ashraful/projects/def-kovens/ashraful/3D_CycleGAN_Tumor_Aware

**If your volumes and masks are under the same project:** input volumes are read from `data_path/labels/` by default (use `--images_subdir images` if your inputs are in `data_path/images/`). Masks from `--mask_dir` must match file count and order.

```bash
python unet_tumor_segmentation/train.py \
  --data_path ./Data_folder/train/ \
  --mask_dir ./Data_folder/train/labels_mask \
  --patch_size 256 256 64 \
  --checkpoint_dir ./unet_tumor_segmentation/checkpoints/mask_range_256_64 \
  --epochs 100 \
  --batch_size 2 \
  --pos_weight 10 \
  --gpu_ids 0
```

- Best model (by Dice on the training set) is saved as:  
  `./unet_tumor_segmentation/checkpoints/best.pth`
- Every `--save_freq` epochs an extra checkpoint is saved (e.g. `epoch_5.pth`).

**Resume from a checkpoint:**

```bash
python unet_tumor_segmentation/train.py \
  --data_path ./Data_folder/train/ \
  --mask_dir ./Data_folder/train/images_mask \
  --checkpoint_dir ./unet_tumor_segmentation/checkpoints \
  --load ./unet_tumor_segmentation/checkpoints/epoch_50.pth \
  --epochs 100 \
  --gpu_ids 0
```

---

## 3. Evaluate (Dice, IoU, HD95)

After training, run evaluation (writes metrics to `checkpoint_dir/metrics.txt`):

```bash
python unet_tumor_segmentation/evaluate.py \
  --load ./unet_tumor_segmentation/checkpoints/best.pth \
  --data_path ./Data_folder/train/ \
  --mask_dir ./Data_folder/train/images_mask \
  --val_data_path ./Data_folder/test/ \
  --val_mask_dir ./Data_folder/test/images_mask \
  --patch_size 128 128 64 \
  --gpu_ids 0
```

If validation = training, you can omit `--val_data_path` / `--val_mask_dir` (they default to train).

---

## 4. Use the pretrained UNet in CycleGAN

Train CycleGAN with the frozen UNet loaded from the pretrained checkpoint:

```bash
python train.py \
  --data_path ./Data_folder/train/ \
  --unet_checkpoint ./unet_tumor_segmentation/checkpoints/best.pth \
  --patch_size 128 128 64 \
  --name my_experiment \
  ... other CycleGAN args ...
```

`--patch_size` in CycleGAN should match what you used for UNet (e.g. `128 128 64`).

---

## Quick reference

| Step        | Command |
|------------|---------|
| **Pretrain UNet** | `python unet_tumor_segmentation/train.py --data_path ... --mask_dir ... --checkpoint_dir ./unet_tumor_segmentation/checkpoints --epochs 100` |
| **Evaluate**      | `python unet_tumor_segmentation/evaluate.py --load ./unet_tumor_segmentation/checkpoints/best.pth --data_path ... --mask_dir ...` |
| **CycleGAN**      | `python train.py --unet_checkpoint ./unet_tumor_segmentation/checkpoints/best.pth ...` |

Replace `--data_path` and `--mask_dir` with your actual paths; keep image and mask lists consistent (same count and order).
