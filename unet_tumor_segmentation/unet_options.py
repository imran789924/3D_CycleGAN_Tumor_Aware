"""
Options for 3D UNet tumor segmentation: train and evaluate with parameterized patch_size.
"""
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='3D UNet tumor segmentation (pretrain for CycleGAN)')
    # Data
    parser.add_argument('--data_path', type=str, default='../Data_folder/train/',
                        help='Base path containing images/ subfolder')
    parser.add_argument('--mask_dir', type=str, default='',
                        help='Directory containing tumor masks (e.g. .../Data_folder/train/images_mask). Same number/order of files as data_path/images.')
    parser.add_argument('--val_data_path', type=str, default='../Data_folder/test/',
                        help='Validation base path (images/ subfolder)')
    parser.add_argument('--val_mask_dir', type=str, default='',
                        help='Validation mask directory (optional; can use mask_dir structure under val_data_path)')
    # Patch size (must match CycleGAN patch_size when loading into CycleGAN)
    parser.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 64],
                        help='Patch size (D, H, W) e.g. 128 128 64')
    parser.add_argument('--new_resolution', type=float, nargs=3, default=[0.45, 0.45, 0.45],
                        help='Resample resolution (if resample=True)')
    parser.add_argument('--resample', action='store_true', help='Resample volumes to new_resolution')
    parser.add_argument('--min_pixel', type=float, default=0.1)
    parser.add_argument('--drop_ratio', type=float, default=0.)
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=5, help='Save checkpoint every N epochs')
    # Checkpoints and output
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save best model and checkpoints (e.g. unet_tumor_segmentation/checkpoints)')
    parser.add_argument('--name', type=str, default='unet_tumor', help='Experiment name')
    parser.add_argument('--load', type=str, default='', help='Load checkpoint to resume or for evaluation')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated GPU ids')
    # Eval-only
    parser.add_argument('--evaluate', action='store_true', help='Only run evaluation (requires --load)')
    args = parser.parse_args()
    # Ensure patch_size is list of int
    args.patch_size = list(map(int, args.patch_size))
    return args
