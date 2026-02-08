"""
Evaluate 3D UNet: Dice, IoU, HD95. Requires --load pointing to checkpoint.
Saves metrics to checkpoint_dir/metrics.txt.
"""
import os
import sys

_unet_dir = os.path.dirname(os.path.abspath(__file__))
if _unet_dir not in sys.path:
    sys.path.insert(0, _unet_dir)
_root = os.path.dirname(_unet_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet_options import parse_args
from dataset import TumorSegDataset
from model import build_unet
from metrics import compute_metrics_batch


def main():
    args = parse_args()
    if not args.load or not os.path.isfile(args.load):
        print('Provide --load path to checkpoint.')
        return

    gpu_ids = [int(x) for x in args.gpu_ids.split(',') if x.strip()]
    device = torch.device('cuda:%d' % gpu_ids[0] if gpu_ids and torch.cuda.is_available() else 'cpu')

    # Validation set: use val_data_path and val_mask_dir if set, else train
    val_data = args.val_data_path if args.val_data_path else args.data_path
    val_mask = args.val_mask_dir if args.val_mask_dir else args.mask_dir
    val_set = TumorSegDataset(
        val_data,
        val_mask,
        patch_size=args.patch_size,
        new_resolution=args.new_resolution,
        resample=args.resample,
        train=False,
    )
    if len(val_set) == 0:
        print('No validation samples. Check val_data_path and val_mask_dir.')
        return
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    ckpt = torch.load(args.load, map_location=device)
    patch_size = ckpt.get('patch_size', args.patch_size)
    model = build_unet(patch_size=patch_size)
    state = ckpt.get('state_dict', ckpt)
    if list(state.keys())[0].startswith('module.'):
        model = nn.DataParallel(model)
    model.load_state_dict(state)
    if hasattr(model, 'module'):
        model = model.module
    model = model.to(device)
    model.eval()

    all_metrics = {'dice': [], 'iou': [], 'hd95': []}
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            m = compute_metrics_batch(pred, mask)
            for k in all_metrics:
                v = m[k]
                if k == 'hd95' and (v != v):
                    continue
                all_metrics[k].append(v)

    mean_dice = sum(all_metrics['dice']) / len(all_metrics['dice']) if all_metrics['dice'] else 0
    mean_iou = sum(all_metrics['iou']) / len(all_metrics['iou']) if all_metrics['iou'] else 0
    hd95_list = [x for x in all_metrics['hd95'] if x == x]
    mean_hd95 = sum(hd95_list) / len(hd95_list) if hd95_list else float('nan')

    results = 'Dice: %.4f  IoU: %.4f  HD95: %.4f\n' % (mean_dice, mean_iou, mean_hd95)
    print(results)

    out_dir = os.path.dirname(args.load)
    metrics_path = os.path.join(out_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(results)
    print('Wrote', metrics_path)


if __name__ == '__main__':
    main()
